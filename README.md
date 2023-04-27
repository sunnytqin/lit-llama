<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/Lit_LLaMA_Badge3x.png" alt="Lit-LLaMA" width="128"/>

# ⚡ Lit-LLaMA-Fork

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![cpu-tests](https://github.com/lightning-AI/lit-llama/actions/workflows/cpu-tests.yml/badge.svg) [![Build Status](https://dev.azure.com/Lightning-AI/lit%20Models/_apis/build/status%2FLightning-AI.lit-LLaMA?branchName=main)](https://dev.azure.com/Lightning-AI/lit%20Models/_build/latest?definitionId=49&branchName=main) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-llama/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

</div>

# ⚡ Lit-LLaMA ️

&nbsp;

## Setup

Clone the repo

```bash
git clone https://github.com/sunnytqin/lit-llama.git 
cd lit-llama
```

install dependencies listed in `requirements.txt`

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions, you don't need to download the model weights. (I think you have reading rights to my folder to access checkpoint weights for lit-llama). If that is causing you a problem, please let me know! 

Run inference:

```bash
python generate.py --model_size 7B
```

This will run the 7B model. The large model is the 30B model by default. 


### Gotchas

Make sure you request 3 GPUs and enough CPU memory (to load the 30B weights). GPU 0 and 1 for large model with pipeline parallelism and GPU 2 for the small model. 

```bash
salloc -p kempner -t 0-02:00 --mem 240000 --gres=gpu:3
```

It takes a couple minutes to load the model but the inference is fast. 

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.

See `python generate.py --help` for more options.


### Example Output

```
(hallucination) [sqin@holygpu8a27606 lit-llama]$ python generate.py --model_size 7B
GPU count:  3
Loading large model ...Time: 199.89 seconds.
Loading small model ...Time: 25.38 seconds.
Type prompt (or 'exit'): Hello, my name is
[Hello, my name is] and(K) I ' m writing you today to learn more about the  2 0 1 6(5) Ford F - 1 5 0 X L(LT) (.) 4 WD Super C rew  5 . 5 '(-) Box available(listed) from(at) North(your) side(Park) Ford(Im) Lincoln(.) in(.) the(Mon) North(Austin) side Ford Lincoln . 

[Hello, my name is] and(0.020) I(0.034) '(0.105) m(0.001) writing(0.017) you(0.005) today(0.004) to(0.014) learn(0.005) more(-0.000) about(-0.000) the(0.001) (0.010) 2(0.001) 0(0.000) 1(0.051) 6(0.198) Ford(0.075) F(0.026) -(0.009) 1(0.008) 5(0.000) 0(0.000) X(0.201) L(0.094) (0.283) 4(0.148) WD(0.005) Super(0.299) C(0.018) rew(0.050) (0.118) 5(0.131) .(0.073) 5(0.059) '(0.578) Box(0.021) available(1.516) from(3.375) North(0.598) side(0.625) Ford(0.385) Lincoln(1.047) in(0.570) the(1.047) North(0.598) side(0.754) Ford(0.492) Lincoln(0.168) .(0.555) 
```

- Small model output is generated autoregressively. Inside () is when large model disagree with the small model 
- It is the top k = 1 prediction. i.e., the token with the highest probability
- The second print inside () is the Jensen Shannon Distance between the large model prediction and small model prediction
