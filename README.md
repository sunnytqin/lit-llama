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

Add this path to your `.condarc`, and activate the environment: 
```
 /n/holylabs/LABS/barak_lab/Lab/sqin/envs/hallucination
```

You are all set! 🎉

&nbsp;

## Use the model

To generate text predictions with the repetition model, 


```bash
python generate.py --repetition=True --model_size 7B
```

This will run the 7B model. See source code for a couple more options for generation. 

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.

To run the 30B model, please request 2 GPUs to avoid OOM, and 3 for 65B. 


To generate text predictions for direct comparison between models, 

```bash
python generate.py --repetition=False --large_model_size=30B
```

This will run the 7B model and 30B model. See source code for a couple more options for generation. 

Please request 3 GPUs for [7B, 30B] and 4 for [7B, 65B] . 
