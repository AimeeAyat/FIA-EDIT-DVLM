
STEP=25
SKIP=5
GUIDANCE=2
EXP_NAME=rf_solver_step_${STEP}_inject_${SKIP}_${GUIDANCE}

export FLUX_SCHNELL=huggingface_models/scheduler
export FLUX_DEV=huggingface_models/flux1-dev.safetensors
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export AE=huggingface_models/ae.safetensors

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 run_rf_edit_pie.py \
        --num_steps $STEP --inject_step $SKIP \
        --guidance $GUIDANCE --exp_name $EXP_NAME

python -m dataset.pie_bench.decode --dir_path pt_result/${EXP_NAME}

python -m evaluation.evaluate --tgt_image_folder output/${EXP_NAME}