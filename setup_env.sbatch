#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=2        # Schedule 8 cores (includes hyperthreading)
#SBATCH --time=03:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu
#SBATCH --partition=scavenge    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END

echo "Running on $(hostname):"
module load Anaconda3
conda create --name brainharmonix
source activate brainharmonix

echo "Installing additional dependencies"
pip install -r requirements-macos.txt

echo "Installing Pytorch with CUDA 11.8"
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"