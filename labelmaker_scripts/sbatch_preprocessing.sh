#!/usr/bin/bash
#SBATCH --job-name="labelmaker-mix3d"
#SBATCH --output=labelmaker_mix3d_preprocessing_%j.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=36

cd /cluster/project/cvg/labelmaker/labelmaker-mix3d
source labelmaker_scripts/activate_env.sh
DATA_DIR="/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker"
SAVE_DIR="data/processed/arkitscenes_labelmaker"

python -m mix3d.datasets.preprocessing.arkitscenes_labelmaker preprocess \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR"
