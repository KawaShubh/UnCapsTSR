#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=100:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
module purge
module --ignore_cache load "Anaconda3/2022.10"
conda env create -f 'dusgan.yml'
conda activate dusgan

# Define the path to the options file
OPTIONS_PATH="/cluster/home/shubhdk/RGT/options/train/train_ntireEx.json"

# Run the Python training script with the specified options
python train.py -opt "$OPTIONS_PATH"
