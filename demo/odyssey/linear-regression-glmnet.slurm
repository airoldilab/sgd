#!/bin/bash
#SBATCH -n 1               # (Max) number of tasks per job, for R usually 1
#SBATCH -o dump/out-linear-regression-glmnet-%a.txt # File for the standard output
#SBATCH -e dump/err-linear-regression-glmnet-%a.txt # File for the standard error
#SBATCH -p stats           # Partition to use
#SBATCH --mem-per-cpu=16384 # Memory required per CPU, in MegaBytes
#SBATCH -a 1               # Array of n jobs
#SBATCH --mail-user=dtran@g.harvard.edu # Where to send mail
#SBATCH --mail-type=ALL    # When to send mail

Rscript linear-regression-glmnet.R

# vim:filetype=sh
