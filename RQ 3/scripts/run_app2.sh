#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=128GB

papermill "RQ 3/Application2.ipynb" app2_out.ipynb
