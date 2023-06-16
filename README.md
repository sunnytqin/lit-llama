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

## Use the GUI

![](https://github.com/sunnytqin/lit-llama/blob/main/demo%20(1).gif)

### To run the GUI

```
python awesomegui.py --data [path_to_LLM_output]
``` 

For a sample output, use `output/sample_output`

You only need a basic python environment (`python 3 + numpy`) to run the GUI - no need to install the entire environment! 

### Output specs

- It is the deterministic top k = 1 prediction. i.e., the token with the highest probability
- For now, we generate 50 new tokens given the prompt auto regressively. I will soon run some teacher-forcing samples
- We display the small model output by default and you need to click the token to see details 

### Gotchas

Make sure you request 3 GPUs and enough CPU memory (to load the 30B weights). GPU 0 and 1 for large model with pipeline parallelism and GPU 2 for the small model. 

```bash
salloc -p kempner -t 0-02:00 --mem 240000 --gres=gpu:3
```

It takes a couple minutes to load the model but the inference is fast. 

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.

See `python generate.py --help` for more options.
