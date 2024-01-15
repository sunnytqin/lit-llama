import sys, os
import time

from jsonargparse import CLI
import torch
import torch.nn as nn

from lit_llama import Tokenizer
from lit_llama.model import pipeLLaMA, LLaMA
from lit_llama.utils import EmptyInitOnDevice
from pathlib import Path

from repetition import repetition_experiment, compute_entropy


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

DEVICE= "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# DTYPE = torch.float32



def main(
    repetition: bool=False, 
    model_type: str = "llama", 
    model_size: str = "7B", 
    large_model_size: str = "30B",
    k: int = 10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,
    sample_until_period: bool = False,
    addl_token_limit: int = 100,
):
    """
    Args:
        model_type: "llama", 
        model_size: [7B, 30B],
        k: top k token for repetition,
        tokenizer_path: llama tokenizer,
        checkpoint_path: model checkpoint default if None,
        sample_until_period: whether sample until period,
        addl_token_limit: sample until period hard cutoff,
    """
    if repetition:
        generate_repetition(model_type=model_type, 
                            model_size=model_size,
                            k=k,
                            tokenizer_path=tokenizer_path,
                            checkpoint_path=checkpoint_path,
                            sample_until_period=sample_until_period,
                            addl_token_limit=addl_token_limit)
    else:
        generate_comparison(model_type=model_type, 
                            small_model_size=model_size,
                            large_model_size=large_model_size,
                            k=k,
                            tokenizer_path=tokenizer_path,
                            checkpoint_path=checkpoint_path,)

    return

def generate_repetition(
        model_type: str = "llama", 
        model_size: str = "7B",
        k: int = 10,
        tokenizer_path: str = None,
        checkpoint_path: str = None,
        sample_until_period: bool = True,
        addl_token_limit: int = 100,):

    if(checkpoint_path is None):
        checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{model_size}/lit-llama.pth")

    if(tokenizer_path is None):
        tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if model_size == "7B":
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model ...", end='', file=sys.stderr)
            t0 = time.time()
            model = LLaMA.from_name(model_size)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    else: 
        # Initialize the model
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model... ", end='')
            t0 = time.time()
            model = pipeLLaMA.from_name(model_size)
            partition_schedule = model.partition_schedule
            checkpoint = torch.load(checkpoint_path)
            for key in list(checkpoint.keys()):
                if 'transformer.h' in key:
                    split = key.split('.')
                    split[2] = partition_schedule[int(split[2])]
                    checkpoint[".".join(split)] = checkpoint.pop(key)
            model.load_state_dict(checkpoint, strict=True)
            print(f"Time: {time.time() - t0:.02f} seconds.")

        model.eval()

    small_lm_head = model.lm_head

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    while True: 

        prompt = input("Type prompt (or 'exit'): ")

        if prompt == 'exit':
            break

        prompt = prompt.strip()
        print("Prompt: ", prompt)

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)
        final_text = []
        final_epi_probs = []
        for _ in range(100):
            next_token, epi_probs = autoregressive_repetition(encoded_prompt, tokenizer, model, sample_until_period, addl_token_limit, model_type, small_lm_head)

            # prompt = (prompt + decoded)
            encoded_prompt = torch.cat([encoded_prompt, next_token], dim=-1)
            
            # hacky decoder for sentencepiece to print space correctly
            decoded = tokenizer.processor.decode(next_token.tolist(), out_type='immutable_proto')[0].pieces
            decoded = str(decoded[0].piece)

            if epi_probs is not None:
                print(format_bg(decoded, epi_probs), end="")
            else:
                print(decoded, end="")
            final_text.append(decoded)
            final_epi_probs.append(epi_probs)
        print("\n")
        
        save_pdf = input("Save example? (type yes or no): ")

        if save_pdf == 'yes':
            create_highlighted_pdf(
                prompt, 
                final_text, 
                final_epi_probs,
                repetition=True
            )


    return 

def autoregressive_repetition(encoded_prompt, tokenizer, model, sample_until_period, addl_token_limit, model_type, small_lm_head, k=10):

    original_embed, repetition_embeds = repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                        sample_until_period=sample_until_period, 
                        addl_token_limit=addl_token_limit,
                        verbose=False)
    original_logits = small_lm_head(original_embed.to(DEVICE)).detach()
    original_probs = torch.softmax(original_logits, dim=-1)
    top1 = torch.max(original_probs, dim=-1).indices
    
    if repetition_embeds is None:
        epi_probs = None
    else:
        new_logits = small_lm_head(repetition_embeds.to(DEVICE)).detach()
        new_probs = torch.softmax(new_logits, dim=-1)
        top_repetition = new_probs.max()
        epi_probs = top_repetition.item()

    return torch.tensor([top1]).to(DEVICE)[None, :], epi_probs

def generate_comparison(model_type: str = "llama", 
    small_model_size: str = "7B",
    large_model_size: str = "30B",
    k=10,
    tokenizer_path: str = None,
    checkpoint_path: str = None,):

    """
    Args:
        model_type: "llama", 
        small: [7B, 30B],
        k: top k token to display,
        tokenizer_path: llama tokenizer,
        checkpoint_path: model checkpoint default if None,
    """

    if(checkpoint_path is None):
        small_checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{small_model_size}/lit-llama.pth")
        large_checkpoint_path = Path(f"/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/{large_model_size}/lit-llama.pth")

    if(tokenizer_path is None):
        tokenizer_path = Path("/n/holystore01/LABS/barak_lab/Everyone/checkpoints/checkpoints/lit-llama/tokenizer.model")

    assert small_checkpoint_path.is_file()
    assert large_checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    if small_model_size == "7B":
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model ...", end='', file=sys.stderr)
            t0 = time.time()
            small_model = LLaMA.from_name(small_model_size)
            checkpoint = torch.load(small_checkpoint_path)
            small_model.load_state_dict(checkpoint)
            print(f"Time: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    else: 
        # Initialize the model
        with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
            print("Loading model... ", end='')
            t0 = time.time()
            small_model = pipeLLaMA.from_name(large_model_size)
            partition_schedule = small_model.partition_schedule
            checkpoint = torch.load(checkpoint_path)
            for key in list(checkpoint.keys()):
                if 'transformer.h' in key:
                    split = key.split('.')
                    split[2] = partition_schedule[int(split[2])]
                    checkpoint[".".join(split)] = checkpoint.pop(key)
            small_model.load_state_dict(checkpoint, strict=True)
            print(f"Time: {time.time() - t0:.02f} seconds.")

        small_model.eval()

    with EmptyInitOnDevice(
            device=DEVICE, dtype=DTYPE, quantization_mode=None,
        ):
        print("Loading model... ", end='')
        t0 = time.time()
        large_model = pipeLLaMA.from_name(large_model_size)
        partition_schedule = large_model.partition_schedule
        checkpoint = torch.load(large_checkpoint_path)
        for key in list(checkpoint.keys()):
            if 'transformer.h' in key:
                split = key.split('.')
                split[2] = partition_schedule[int(split[2])]
                checkpoint[".".join(split)] = checkpoint.pop(key)
        large_model.load_state_dict(checkpoint, strict=True)
        print(f"Time: {time.time() - t0:.02f} seconds.")

    large_model.eval()

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    period_id = tokenizer.encode("Period.", bos=False, eos=False, device=DEVICE)[-1].item()

    while True: 

        prompt = input("Type prompt (or 'exit'): ")

        if prompt == 'exit':
            quit()

        prompt = prompt.strip()
        print("Prompt: ", prompt)

        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=DEVICE).unsqueeze(0)
        final_text = []
        final_entropy = []
        for token_cnt in range(150):
            next_token, entropy_diff = autoregressive_comparison(encoded_prompt, small_model, large_model, tokenizer)
            encoded_prompt = torch.cat([encoded_prompt, next_token], dim=-1)

            # hacky decoder for sentencepiece to print space correctly
            decoded = tokenizer.processor.decode(next_token.tolist(), out_type='immutable_proto')[0].pieces
            decoded = str(decoded[0].piece) 

            final_text.append(decoded)
            final_entropy.append(entropy_diff)

            # finish at period
            if token_cnt >= 100:
                if next_token == period_id:
                    break
        print('\n')

        save_pdf = input("Save example? (type yes or no): ")

        if save_pdf == 'yes':
            create_highlighted_pdf(
                prompt, 
                final_text, 
                final_entropy,
                repetition=False
            )
            
    
    return

def autoregressive_comparison(encoded_prompt, small_model, large_model, tokenizer, k=1):
    # print(f"\n{small_model_size} Model: ", end = " ")  
    small_logits = small_model(encoded_prompt.to(DEVICE))[0, -1, :].detach()
    small_entropy = compute_entropy(small_logits)
    small_logits = torch.softmax(small_logits, dim=-1)
    small_top_k = torch.topk(small_logits, k, dim=-1).indices
    small_decoded = [tokenizer.decode(rt) for rt in torch.unbind(small_top_k)]
    prob = [float(small_logits[rt]) for rt in small_top_k]
    # for d, p in zip(small_decoded, prob):
    #     print(f"{d}({p:.2f})", end=" ")
    # print(f"{small_decoded[0]}", end=" ")
    # print(f"(small entropy: {small_entropy:.3f})", end= " ") 
    # print(f"\n{large_model_size} Model: ", end = " ")   

    large_logits = large_model(encoded_prompt.to(DEVICE))[0, -1, :].detach()
    large_entropy = compute_entropy(large_logits)
    large_logits = torch.softmax(large_logits, dim=-1)
    large_top_k = torch.topk(large_logits, k, dim=-1).indices
    large_decoded = [tokenizer.decode(rt) for rt in torch.unbind(large_top_k)]

    if small_entropy > 0.0:
        entropy_diff = torch.abs(small_entropy - large_entropy)
        print(format_bg_comparison(small_decoded[0], entropy_diff), end=" ")
    else:
        print(f"{small_decoded[0]}", end=" ")
        entropy_diff = None
        

    # if torch.abs(small_entropy - large_entropy) > 1.0:
    #     prob = [float(large_logits[rt]) for rt in large_top_k]
    #     for d, p in zip(large_decoded, prob):
    #         print(f"({d}: {p:.2f})", end=" ")
    #     print(f"({small_entropy:.2f} {large_entropy:.2f})") 

    return torch.tensor([small_top_k]).to(DEVICE)[None, :], entropy_diff

def format_bg(text, epi_probs):
        assert (0. <= epi_probs) and (epi_probs <= 1.)
        epi_probs = int(epi_probs * 100) // 20
        color_dict = {0: 154, 1: 148, 2: 172, 3: 166, 4: 160, 5:124}
        color_formatter = "\33[38;5;" + str(color_dict[epi_probs]) + "m" + text + "\33[0m"
        return color_formatter

def format_bg_comparison(text, entropy_diff):
        entropy_bins = [0, 0.2, 0.5, 1.0, 2.0, 1000.]
        color_dict = {0: 154, 1: 148, 2: 172, 3: 166, 4: 160}
        for i in range(len(entropy_bins)):
            if entropy_diff >= entropy_bins[i] and entropy_diff <= entropy_bins[i+1]:
                color_formatter = "\33[38;5;" + str(color_dict[i]) + "m" + text + "\33[0m"
                return color_formatter
    # color scheme reference: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
            
def print_comparison_scheme():
    text = ['[0, 0.2)', '[0.2, 0.5)', '[0.5, 1.0)', '[1.0, 2.0)', '[2.0, inf)']
    entropy_diff = [0.1, 0.3, 0.7, 1.2, 3.]
    print("Entropy difference color scheme: ")
    for i in range(len(entropy_diff)):
        print(format_bg_comparison(text[i], entropy_diff[i]), end=' ')
    print("\n")

def print_comparison_scheme_pdf():
    prompt = "entropy difference: "
    text = ['[0, 0.2)', '[0.2, 0.5)', '[0.5, 1.0)', '[1.0, 2.0)', '[2.0, inf)']
    entropy_diff = [0.1, 0.3, 0.7, 1.2, 3.]
    create_highlighted_pdf(
    prompt, 
    text, 
    entropy_diff,
    output_path = "example_text.pdf",
    repetition = False,)

def print_repetition_scheme():
    text = ['[0, 0.2)', '[0.2, 0.5)', '[0.5, 1.0)', '[1.0, 2.0)', '[2.0, inf)']
    entropy_diff = [0.1, 0.3, 0.7, 1.2, 3.]
    print("Entropy difference color scheme: ")
    for i in range(len(entropy_diff)):
        print(format_bg_comparison(text[i], entropy_diff[i]), end=' ')
    print("\n")


def create_highlighted_pdf(
    prompt, 
    generation, 
    epi_probs,
    output_path = "example_text.pdf",
    skip_after = None,
    repetition = False,
):
    X_MARGIN = 25
    X_WIDTH = 600
    Y_MARGIN = 25
    Y_INTERVAL = 15
    if repetition:
        color_scheme = []
        for i in range(5):
            color_scheme.append((1 - 0.2 * i, 0.2 * i, 0.3))
        color_scheme.append((1, 0, 0.3))

    else:
        color_scheme = []
        color_scheme = [(0, 110/255, 46/255), # green 
                        (153/255, 204/255, 51/255),
                        (204/255, 102/255, 51/255), #orange
                        (204/255, 51/255, 51/255),
                        (1, 0, 0) # red
                        ]
        entropy_bins = [0, 0.2, 0.5, 1.0, 2.0, 1000.]
        
    # Create a PDF document with a fixed page size
    pdf_canvas = canvas.Canvas(output_path, pagesize=letter)

    # Set font and size
    font_name = "Courier"
    font_size = 12
    pdf_canvas.setFont(font_name, font_size)

    # Figure out the size of the text
    x = X_MARGIN
    line_count = 1
    used_line = False
    word_lengths = []
    prompt = "Prompt: " + prompt + " <0x0A> <0x0A> Generated: "
    prompt_array = prompt.split(" ")
    # add space back
    for i in range(len(prompt_array)):
        if prompt_array[i] != "<0x0A>": # new line
            prompt_array[i] += " " 
    # re-adjust space for generated tokens
    if generation[0][0] == '▁':
        generation[0] = generation[0].replace('▁', '')
    for i in range(len(generation) - 1):
        if generation[i+1][0] == '▁':
            generation[i+1] = generation[i+1].replace('▁', '')
            generation[i] += " "

    epi_probs = [None] * len(prompt_array) + epi_probs
    for word in prompt_array + generation:
        if word == '<0x0A>': # new line
            x = X_MARGIN
            line_count += 1
            used_line = True

        else: # Move to the next position
            word_length = pdf_canvas.stringWidth(word, font_name, font_size)
            x += word_length
            word_lengths.append(word_length)
            used_line = True

            # Check if a new line is needed
            if x >= X_WIDTH + X_MARGIN:
                x = X_MARGIN
                line_count += 1
                used_line = True

    if(not used_line):
        line_count -= 1

    # Resize the canvas
    pdf_canvas.setPageSize((
        X_WIDTH + X_MARGIN, 
        (line_count) * Y_INTERVAL + 2*Y_MARGIN
    ))

    # Set the initial position for drawing
    x, y = X_MARGIN, line_count * Y_INTERVAL + Y_MARGIN

    # Iterate through words and draw them on the PDF
    # token_idx = 0
    for token_idx, word in enumerate(prompt_array + generation):
        cursor = 0
        word_length = pdf_canvas.stringWidth(word, font_name, font_size)
        # Skip a line if the next word is long
        if(x + word_length >= X_WIDTH + X_MARGIN):
            x = X_MARGIN
            y -= Y_INTERVAL
        if word == '<0x0A>': # new line
            x = X_MARGIN
            y -= Y_INTERVAL
            token_idx += 1
            continue 
        
        if repetition:
            if epi_probs[token_idx] is not None:
                idx = int((epi_probs[token_idx] * 100) // 20)
                pdf_canvas.setFillColorRGB(*color_scheme[idx])  # Red color
            else:
                pdf_canvas.setFillColorRGB(0, 0, 0)  # Black color
        else:
            if epi_probs[token_idx] is not None:
                pdf_canvas.setFillColorRGB(0, 0, 0)  # Black color defaulxt
                for i in range(len(entropy_bins)-1):
                    if epi_probs[token_idx] >= entropy_bins[i] and epi_probs[token_idx] <= entropy_bins[i+1]:
                        pdf_canvas.setFillColorRGB(*color_scheme[i])  # Red color
            else:
                pdf_canvas.setFillColorRGB(0, 0, 0)  # Black color
                    

        pdf_canvas.drawString(x, y, word)

        x += pdf_canvas.stringWidth(word, font_name, font_size)
        cursor += len(word)
        # token_idx += 1

    # Save the PDF
    pdf_canvas.save()
    return

if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)

    # print_comparison_scheme_pdf()

