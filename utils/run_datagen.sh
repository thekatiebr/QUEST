#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=48:00:00


module load cuda10.1
module load cudnn

python utils/GenerateUncertaintyData.py ${1}
# python RQ\ 1/VarianceMapper.py NN-dropout
# python RQ\ 1/VarianceMapper.py catboost-ve