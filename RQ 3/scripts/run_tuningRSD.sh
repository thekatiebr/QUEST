#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=128GB

papermill "RQ 3/RSD_TradeoffCurves.ipynb" "output/RSD_TradeoffCurves.ipynb"