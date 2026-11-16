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

echo "Installing Pytorch with CUDA 11.8"
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing additional dependencies"
pip install -r requirements-macos.txt
