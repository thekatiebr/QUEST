sbatch RQ\ 3/run_subgroupdisc.sh trauma_uk NN-dropout
sbatch RQ\ 3/run_subgroupdisc.sh trauma_uk catboost-ve

sbatch RQ\ 3/run_subgroupdisc.sh diabetes catboost-ve

sbatch RQ\ 3/run_subgroupdisc.sh critical_outcome catboost-ve
# sbatch RQ\ 3/run_subgroupdisc.sh critical_triage catboost-ve
sbatch RQ\ 3/run_subgroupdisc.sh ED_3day_readmit catboost-ve
sbatch RQ\ 3/run_subgroupdisc.sh hospitalization_prediction catboost-ve

sbatch RQ\ 3/run_subgroupdisc.sh diabetes NN-dropout
sbatch RQ\ 3/run_subgroupdisc.sh critical_outcome NN-dropout
# sbatch RQ\ 3/run_subgroupdisc.sh critical_triage NN-dropout
sbatch RQ\ 3/run_subgroupdisc.sh ED_3day_readmit NN-dropout
sbatch RQ\ 3/run_subgroupdisc.sh hospitalization_prediction NN-dropout
