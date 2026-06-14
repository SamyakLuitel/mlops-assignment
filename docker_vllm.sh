#!/usr/bin/env bash

docker run --gpus all -v ./infra:/infra -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
        --ipc=host \
     vllm/vllm-openai:latest --config /infra/vllm_config.yaml