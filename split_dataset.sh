#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=split-dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/cchen795/scratch/slurm/%x-%j.out
#SBATCH --error=/home/cchen795/scratch/slurm/%x-%j.out

echo "Loading modules"
source $HOME/scratch/py36init.sh
source cc.env.sh

python split_dataset.py \
	-v \
	--data-files $(ls out/*.pkl) \
	--n-splits 3 \
	--n-val 60 \
	--n-test 60 \
	--n-train 800 \
	--label v3-2-1

