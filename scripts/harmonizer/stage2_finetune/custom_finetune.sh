#!/bin/bash

#SBATCH --job-name=brainharmonix-embed
#SBATCH --output=logs/finetune.%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=acltr
#SBATCH --exclude=cn[8,14-17],desktop[1,6-8,12,15],asus1,dgx1
#SBATCH --mail-type=BEGIN,FAIL,END

echo "Activating Environment"
module load Anaconda3
source activate brainharmonix

# Run
echo "Run finetuning"
export PYTHONPATH="$PWD"
accelerate launch modules/harmonizer/stage2_finetune/main_finetune.py \
    --dataset_name AbideI \
    --nb_classes 2 \
    --split_seed 0 \
    --batch_size 16 \
    --num_workers 4 \
    --epochs 50 \
    --output_dir experiments/stage2_finetune/AbideI \
    --log_dir experiments/stage2_finetune/AbideI
