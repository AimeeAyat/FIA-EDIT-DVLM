export FLUX_SCHNELL=huggingface_models/scheduler
export FLUX_DEV=huggingface_models/flux1-dev.safetensors
export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export AE=huggingface_models/ae.safetensors
python gradio_kv_edit_gpu.py --gpus