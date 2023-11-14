#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=32GB

echo ${1}
echo ${2}

python RQ\ 3/categorical_rsd.py ${1} ${2}