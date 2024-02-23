env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_arkitscenes/2024-02-19_1444/1b9abe1d/20419e2c_5d34/switch_cls_head\=20.ckpt"

cd /home/guangda/repos/labelmaker-mix3d
poetry run train --config-name="config_scannet20.yaml" \
    general.checkpoint="$CKPT_PATH" \
    general.freeze_backbone="true" \
    trainer.check_val_every_n_epoch="1" \
    callbacks.0.period="1" \
    data.num_labels="20"
