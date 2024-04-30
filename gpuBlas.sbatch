#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --account=mpcs51087
#SBATCH --partition=gpu
#SBATCH --job-name=gpuBLAS
#SBATCH --output=gpuBLAS.out
#SBATCH --error=gpuBLAS.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user==jbrouwer@uchicago.edu
#SBATCH --exclusive

./cublas 1 800 50 200 .1