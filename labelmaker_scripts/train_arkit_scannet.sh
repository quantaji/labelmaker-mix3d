#!/usr/bin/bash
#SBATCH --job-name="arkit-scannet"
#SBATCH --output=arkit_scannet_train_%j.out
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH -A ls_polle
#SBATCH --gpus=a100_80gb:1

module purge
module load gcc/11.4.0 cuda/12.1.1 eth_proxy

export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"
env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

source_dir=/cluster/project/cvg/labelmaker/labelmaker-mix3d
target_dir=$TMPDIR/labelmaker-mix3d
echo "Start coping files!"
rsync -r \
    --exclude="saved" \
    $source_dir/ \
    $target_dir
echo "Files copy finished!"

cd $target_dir
poetry run train --config-name="config_arkit_scannet200.yaml"
