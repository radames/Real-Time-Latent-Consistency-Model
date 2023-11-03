---
title: Real-Time Latent Consistency Model Image-to-Image
emoji: 🖼️🖼️
colorFrom: gray
colorTo: indigo
sdk: docker
pinned: false
suggested_hardware: a10g-small
---

# Real-Time Latent Consistency Model

This demo showcases [Latent Consistency Model (LCM)](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) using [Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/community#latent-consistency-pipeline) with a MJPEG stream server.

You need a webcam to run this demo. 🤗

## Running Locally

You need CUDA and Python 3.10, Mac with an M1/M2/M3 chip or Intel Arc GPU

`TIMEOUT`: limit user session timeout  
`SAFETY_CHECKER`: disabled if you want NSFW filter off   
`MAX_QUEUE_SIZE`: limit number of users on current app instance  

### image to image

```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload
```

### text to image

```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
uvicorn "app-txt2img:app" --host 0.0.0.0 --port 7860 --reload
```

or with environment variables

```bash
TIMEOUT=120 SAFETY_CHECKER=True MAX_QUEUE_SIZE=4 uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload
```

If you're running locally and want to test it on Mobile Safari, the webserver needs to be served over HTTPS.

```bash
openssl req -newkey rsa:4096 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem
uvicorn "app-img2img:app" --host 0.0.0.0 --port 7860 --reload --log-level info --ssl-certfile=certificate.pem --ssl-keyfile=key.pem
```

## Docker

You need NVIDIA Container Toolkit for Docker

```bash
docker build -t lcm-live .
docker run -ti -p 7860:7860 --gpus all lcm-live
```

or with environment variables

```bash
docker run -ti -e TIMEOUT=0 -e SAFETY_CHECKER=False -p 7860:7860 --gpus all lcm-live
```

# Demo on Hugging Face

https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model

https://github.com/radames/Real-Time-Latent-Consistency-Model/assets/102277/c4003ac5-e7ff-44c0-97d3-464bb659de70
