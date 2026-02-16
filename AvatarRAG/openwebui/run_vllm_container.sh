
  $env:HF_TOKEN="your hf token"
docker run --gpus all -v C:\Users\Dhaya\.cache\huggingface:/root/.cache/huggingface -e HF_TOKEN=$env:HF_TOKEN -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.2-3B-Instruct --quantization bitsandbytes --dtype float16 --max-model-len 512 --max-num-seqs 6 --gpu-memory-utilization 0.85


docker run -d --name openwebui -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=dummy -e CUDA_VISIBLE_DEVICES="" --restart unless-stopped ghcr.io/open-webui/open-webui:main