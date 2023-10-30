---
title: Real-Time Latent Consistency Model
emoji: 🔥
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
---

# Real-Time Latent Consistency Model

This demo showcases [Latent Consistency Model (LCM)](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using [Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community#latent-consistency-pipeline) with a MJPEG stream server.

## Running Locally

You need CUDA and Python  
`TIMEOUT`: limit user session timeout
`SAFETY_CHECKER`:  disabled if you want NSFW filter off  

```bash
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt
TIMEOUT=0 SAFETY_CHECKER=False uvicorn "app:app" --host 0.0.0.0 --port 7860 --reload
```

## Docker
You need NVIDIA Container Toolkit for Docker

```bash
docker build -t lcm-live .
docker run -ti -e TIMEOUT=0 -e SAFETY_CHECKER=False -p 7860:7860 --gpus all lcm-live
```
# Demo on Hugging Face
https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model

![video](https://github.com/radames/Real-Time-Latent-Consistency-Model/assets/102277/2fb8336c-62b3-4aac-97a7-f6c1fac4f38b)
