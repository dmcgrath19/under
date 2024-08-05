#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N rkvw-s
#$ -o /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=300G
#$ -l h_rt=24:00:00
#$ -m bea -M s2558433@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

export CXXFLAGS="-std=c99"
export CFLAGS="-std=c99"
export TOKENIZERS_PARALLELISM=false

. /etc/profile.d/modules.sh
module unload cuda

module load cuda/12.1.1
#qlogin -q gpu -pe gpu-a100 1 -l h_vmem=500G -l h_rt=24:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda

cd /exports/eddie/scratch/s2558433/under/

python run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/pythia-2.8b --mask_filling_model_name t5-3b --n_perturbation_list 25 --n_samples 2000 --pct_words_masked 0.3 --span_length 2 --cache_dir cache --dataset_member wiki --dataset_member_key text --dataset_nonmember wmt  --max_length 2000

conda deactivate 
