#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=16GB

echo ${1}
echo ${2}

python RQ\ 3/continuous_rsd.py ${1} ${2}