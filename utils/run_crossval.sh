#!/bin/bash
#SBATCH --mem=32gb

<<<<<<< HEAD
# python utils/cross_validation.py
papermill "utils/cross_validate.ipynb" "output/cross_validate.ipynb"
=======
python utils/cross_validation.py
# papermill ux/tils/cross_validate.ipynb
>>>>>>> 96a5248679208217495f79d9e0fc831b1e972e30
