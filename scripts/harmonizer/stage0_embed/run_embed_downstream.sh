#!/bin/bash

#SBATCH --job-name=brainharmonix-embed
#SBATCH --output=logs/embed.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=acltr
#SBATCH --exclude=cn[8,14-17],desktop[1,6-8,12,15],asus1,dgx1

# Get parameters
config_file=$1

if [[ ! -f "$config_file" ]]; then
    echo "❌ Error: Configuration file '$config_file' does not exist!"
    exit 1
fi

# Display configuration
echo "=== BrainHarmonix Stage0 Embedding Extraction ==="
echo "Configuration file: $config_file"
echo "=== Starting Training ==="

echo "Activating Environment"
module load Anaconda3
source activate brainharmonix

# Run
accelerate launch \
    --main_process_port 9999 \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    modules/harmonizer/stage0_embed/embedding_downstream.py \
    --config=${config_file}
