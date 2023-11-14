#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
module load cuda10.1
module load cudnn

papermill "utils/ImageCNN.ipynb" "output/ImageCNN.ipynb"
