#!/bin/bash

#SBATCH --account=ds_ga_1012-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=truthfulqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ab12057@nyu.edu
#SBATCH --output="truthfulqa.out"


cd /scratch/ab12057/dsga1012/hw3
conda init && conda activate dsga1012-hw3
bash /scratch/ab12057/dsga1012/hw3/run_truthfulqa_evaluation.sh