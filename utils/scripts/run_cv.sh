#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=16GB

conda activate py36
dataset=${1}
model=${2}

python utils/cross_validate.py ${dataset} NeuralNet
python utils/cross_validate.py ${dataset} CatBoost
python utils/cross_validate.py ${dataset} RandomForest