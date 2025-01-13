#!/bin/bash

# example usage
# sh scripts/finetune_with_accelerate_config_sweep.sh 1 configs/train_configs/sft/default.yaml
# sh scripts/finetune_with_accelerate_config_sweep.sh 8 configs/train_configs/sft/olmo_17_sft.yaml

# Exit immediately if a command exits with a non-zero status
set -e

# Check if exactly three arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <num_gpus> <config_file> <wandb_project_name> <model_name>"
    echo "Example: $0 2 path/to/config.yaml my_wandb_project <model_name>"
    exit 1
fi

NUM_GPUS="$1"
CONFIG_FILE="$2"
WANDB_PROJECT_NAME="$3"

# Generate CUDA_VISIBLE_DEVICES as a range from 0 to NUM_GPUS-1
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES

echo "Number of GPUs: $NUM_GPUS"
echo "Using config file: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Define an array of learning rates to sweep over
LEARNING_RATES="5e-5 1e-4 5e-4 1e-3"

for LR in $LEARNING_RATES; do
    echo "Running with learning rate: $LR"
    
    # You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory, 
    # but it will trade off speed.
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        open_instruct/finetune.py \
        "$CONFIG_FILE" \
        --learning_rate=$LR \
        --wandb_project_name=$WANDB_PROJECT_NAME \
        --run_name="lr=$LR" \
        --model_name_or_path=$4 \
        --tokenizer_name=$4
        #--report_to=tensorboard,wandb
done