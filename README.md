# Hermes RunPod Serverless Worker

Custom load-balanced RunPod Serverless worker for the Hermes Telegram agent.
It preserves the existing OpenAI-compatible `llama.cpp` API and exact
Qwen3.6-27B GGUF runtime configuration while allowing the GPU worker count to
scale to zero.

## Runtime contract

- Main port: `$PORT` (default `80`)
- Health: `GET /ping`
  - `204` while the model is loading
  - `200` when `/v1/models` is ready
  - `503` after an initialization failure
- Diagnostics: `GET /worker/status`
- Model API: `/v1/*`
- Model cache:
  `oolfBER/Qwen3.6-27B-UD-Q4_K_XL-single-GGUF`
- Model filename: `Qwen3.6-27B-UD-Q4_K_XL.gguf`

The worker deliberately does not log prompts or authorization headers.
External authentication is provided by the RunPod endpoint.

## Pinned components

- CUDA: 12.4.1
- Ubuntu: 22.04
- `llama.cpp`: `3018a11e79e489b657dbb77c95694889ccff92df`
- Model byte size: `17909097600`

The image does not contain the GGUF. RunPod's cached-model mount supplies it
under `/runpod-volume/huggingface-cache/hub`.

## Local container test

The full model needs an NVIDIA GPU. With a model mounted explicitly:

```bash
docker run --rm --gpus all \
  -p 8088:80 \
  -e MODEL_PATH=/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -v /path/to/models:/models:ro \
  ghcr.io/geroolf/hermes-runpod-serverless:VERSION
```

## Deployment defaults

- Endpoint type: Load Balancer
- Active workers: 0
- Maximum workers: 1
- Idle timeout: 180 seconds
- FlashBoot: enabled
- GPUs: 24 GB value class with tested fallback categories
- Health check: `/ping`
- Cached model:
  `oolfBER/Qwen3.6-27B-UD-Q4_K_XL-single-GGUF`

