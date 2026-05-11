STEP=28
GUIDANCE=1.5
SKIP=0
export FLUX_SCHNELL=huggingface_models/scheduler
export FLUX_DEV=huggingface_models/flux1-dev.safetensors
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export AE=huggingface_models/ae.safetensors

EXP_NAME=reconstruction_step_${STEP}_skip_${SKIP}_${GUIDANCE}_first_order
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 run_reconstruction.py \
    --num_steps $STEP --skip_step $SKIP \
    --guidance $GUIDANCE --exp_name $EXP_NAME
python -m dataset.pie_bench.decode --dir_path pt_result/${EXP_NAME}