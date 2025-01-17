#!/bin/bash
#SBATCH --job-name=spectra_2b_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1               
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=spectra_2b_sft_%j.out    # stdout (%j will be replaced by the job ID)
#SBATCH --error=spectra_2b_sft_%j.err 
#SBATCH --partition=long  # Replace with your actual partition name
#SBATCH --requeue                        # Enable requeueing if the job is preempted
#SBATCH --gres=gpu:a100l:8

# Move to the correct directory
cd /home/mila/a/ayush.kaushal/scratch/pshishodia/open-instruct 

# Activate Conda environment
source /home/mila/a/ayush.kaushal/miniconda3/etc/profile.d/conda.sh # conda info --base
conda activate .conda-env

# Load CUDA module
module load cuda/12.1.1

# Run your finetuning script with the configuration file
sh /home/mila/a/ayush.kaushal/scratch/pshishodia/open-instruct/scripts/finetune_with_accelerate_config.sh 8 /home/mila/a/ayush.kaushal/scratch/pshishodia/open-instruct/configs/train_configs/spectra/spectra2_1b_sft.yaml