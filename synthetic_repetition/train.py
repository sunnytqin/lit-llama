import argparse
import math
import os
import time

import torch
from torch.distributed import (
    init_process_group, 
    destroy_process_group,
)
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from data import (
    compute_epistemic_collision_prob,
    SyntheticRepetitionDataset,
    SyntheticRepetitionTokenizer
)
from model import (
    GPT,
    GPTConfig,
)

DTYPE = torch.float16
assert(torch.cuda.is_available())
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul

# WandB x axis
WANDB_STEP_METRICS = set(["step", "epoch"])
# WandB y, x pairs
wandb_metrics = set([
    ("val/loss", "step"),
    ("mfu", "step"),
    ("lr", "step"),
])


def _wandb_setup(args):
    wandb_args = {
        "project": "wandb_project",
        "entity": "wandb_entity",
        "name": "wandb_run_name",
        "dir": "output_dir",
    }
    for arg in wandb_args.values():
        if(args[arg] is None):
            raise ValueError(f"Must provide {arg} if use_wandb is True")

    wandb.login()

    wandb_config = {k:v for k,v in args.items()}
    slurm_jobid = os.environ["SLURM_JOBID"]
    if(slurm_jobid):
        wandb_config["slurm_jobid"] = slurm_jobid

    wandb_run = wandb.init(
        config=wandb_config,
        **{k:args[v] for k, v in wandb_args.items()},
    )

    for step_metric in WANDB_STEP_METRICS:
        wandb.define_metric(step_metric)

    for metric, step_metric in wandb_metrics:
        assert(step_metric in WANDB_STEP_METRICS)
        wandb.define_metric(metric, step_metric=step_metric)

    # Save the git diff for reproducibility
    git_diff_path = os.path.join(args[wandb_args["dir"]], "git_diff.txt")
    os.system(f"git diff > {git_diff_path}")
    wandb.save(git_diff_path, base_path=f"./{args[wandb_args['dir']]}")

    return wandb_run


def main(args):
    # Print some stats about the current run
    epistemic_collision_prob = compute_epistemic_collision_prob(
        question_length=args.question_length,
        epistemic_prob=args.epistemic_prob,
        questions_per_sample=args.questions_per_sample,
    )
    print(f"Epistemic collision probability: {epistemic_collision_prob}")
    print(f"Per batch: {1 - (1 - epistemic_collision_prob) ** args.batch_size}")
    print(f"Seqlen: {args.questions_per_sample * (args.question_length + 1 + 1)}")

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps = args.gradient_accumulation_steps // ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda'
        gradient_accumulation_steps = args.gradient_accumulation_steps

    ctx = torch.amp.autocast(device_type=device, dtype=DTYPE)

    tokenizer = SyntheticRepetitionTokenizer()

    dataset = SyntheticRepetitionDataset(
        question_length=args.question_length,
        epistemic_prob=args.epistemic_prob,
        questions_per_sample=args.questions_per_sample,
        force_collision_prob=args.force_collision_prob,
        seed=args.seed + seed_offset,
        val=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    val_dataset = SyntheticRepetitionDataset(
        question_length=args.question_length,
        epistemic_prob=args.epistemic_prob,
        questions_per_sample=args.questions_per_sample,
        force_collision_prob=args.force_collision_prob,
        seed=args.seed + ddp_world_size + 1,
        val=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
    )

    # question + answer + eos
    block_size = (args.question_length + 1 + 1) * args.questions_per_sample
    model_args = dict(
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        block_size=block_size,
        bias=args.bias, 
        vocab_size=64, # 3 rounded up to the nearest multiple of 64
        dropout=args.dropout
    )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    pcount = sum(p.numel() for p in model.parameters())
    print(f"Model initialized ({pcount} params)...")

    optimizer = model.configure_optimizers(
        args.weight_decay, 
        args.learning_rate, 
        (args.beta1, args.beta2), 
        'cuda',
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # compile the model
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

    if(ddp):
        model = DDP(model, device_ids=[ddp_local_rank])

    def postprocess_batch(batch):
        questions, answers = batch
        question_length = len(questions[0][0])
        prompts = []
        for question_set, answer_set in zip(zip(*questions), zip(*answers)):
            qa_tokens = [tokenizer.encode(f"{q}{a}") for q, a in zip(question_set, answer_set)]
            prompt = []
            for qa in qa_tokens:
                prompt.extend(qa)
                prompt.append(tokenizer.eos_token_id)

            prompts.append(prompt)    

        batch = torch.tensor(prompts, dtype=torch.long, device=device)

        targets = batch[..., 1:].contiguous()

        # We only care about the answer bits
        masked_targets = targets.new_zeros(targets.shape)
        masked_targets = masked_targets - 1
        for i in range(question_length - 1, targets.shape[-1], question_length + 1 + 1):
            masked_targets[..., i] = targets[..., i]
            assert(torch.all(targets[..., i + 1] == tokenizer.eos_token_id))

        batch = batch[..., :-1].contiguous()

        return batch, masked_targets

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = torch.zeros(args.eval_iters)
        for k, batch in enumerate(val_dataloader):
            if(k == args.eval_iters):
                break
            with ctx:
                batch, targets = postprocess_batch(batch)
                logits, loss = model(batch, targets)
            losses[k] = loss.item()
        model.train()
        return losses.mean()
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)
   
    # logging
    if args.use_wandb and master_process:
        import wandb
        _wandb_setup(vars(args))

    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    best_val_loss = 1e9
    t0 = time.time()
    for batch_no, batch in enumerate(dataloader):
        # determine and set the learning rate for this iteration
        lr = get_lr(batch_no) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if batch_no % args.eval_interval == 0 and master_process:
            val_loss = estimate_loss()
            print(f"step {batch_no}: val loss {val_loss:.4f}")
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "step": batch_no,
                })
            if val_loss < best_val_loss or args.always_save_checkpoint:
                best_val_loss = val_loss
                if batch_no > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': batch_no,
                        'best_val_loss': best_val_loss,
                        'config': args,
                    }
                    print(f"saving checkpoint to {args.output_dir}")
                    torch.save(checkpoint, os.path.join(args.output_dir, 'ckpt_heavy.pt'))

            if batch_no == 0 and args.eval_only:
                break

        batch, targets = postprocess_batch(batch)

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(batch, targets)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # backward pass, with gradient scaling if training in fp16
            loss.backward()
        # clip the gradient
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if batch_no % args.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(args.batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            print(f"iter {batch_no}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2E}, mfu {running_mfu*100:.2f}%")


        local_iter_num += 1

        # termination conditions
        if batch_no > args.max_iters:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_length", type=int, default=19)
    parser.add_argument("--epistemic_prob", type=float, default=0.5)
    parser.add_argument("--questions_per_sample", type=int, default=15)
    parser.add_argument("--force_collision_prob", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--bias", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--decay_lr", type=bool, default=True)
    parser.add_argument("--lr_decay_iters", type=int, default=600000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=600000)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--always_save_checkpoint", type=bool, default=True)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)
