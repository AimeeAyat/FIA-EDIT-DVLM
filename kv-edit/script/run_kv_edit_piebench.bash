
STEP=28
SKIP=4
INVERSION_GUIDANCE=1.5
DENOISE_GUIDANCE=5.5
EXP_NAME=step_${STEP}_skip_${SKIP}_${INVERSION_GUIDANCE}_${DENOISE_GUIDANCE}

export FLUX_SCHNELL=huggingface_models/scheduler
export FLUX_DEV=huggingface_models/flux1-dev.safetensors
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export AE=huggingface_models/ae.safetensors
export OMP_NUM_THREADS=8

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 run_kv_edit_pie.py \
        --inversion_num_steps $STEP --denoise_num_steps $STEP --skip_step $SKIP \
        --inversion_guidance $INVERSION_GUIDANCE --denoise_guidance $DENOISE_GUIDANCE --exp_name $EXP_NAME

python -m dataset.pie_bench.decode --dir_path pt_result/${EXP_NAME}

python -m evaluation.evaluate --tgt_image_folder output/${EXP_NAME}