#!/usr/bin/bash
#SBATCH --job-name="scannet"
#SBATCH --output=scannet_train_%j.out
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH -A ls_polle
#SBATCH --gpus=rtx3090:1

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
    --exclude="data/processed/arkitscenes_labelmaker" \
    --exclude="data/processed/scannet200" \
    --exclude="saved" \
    $source_dir/ \
    $target_dir
echo "Files copy finished!"

eval_period="1"
save_period="1"
freeze_backbone="true"
# freeze_backbone="false"
CKPT_PATH="/cluster/project/cvg/labelmaker/labelmaker-mix3d/saved/baseline_arkitscenes/2024-02-19_1444/1b9abe1d/20419e2c_5d34/switch_cls_head\=20.ckpt"

cd $target_dir
poetry run train --config-name="config_scannet20.yaml" \
    general.checkpoint="$CKPT_PATH" \
    general.freeze_backbone=$freeze_backbone \
    trainer.check_val_every_n_epoch=$eval_period \
    callbacks.0.period=$save_period \
    data.num_labels="20"
