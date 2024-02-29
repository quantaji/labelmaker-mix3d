env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_scannet/2024-02-24_0528/5fa5d3bd/54180504_e89d/epoch\=487-val_IoU_0\=0.741.ckpt"

TEST_MODE="validation"
# TEST_MODE="test"
cd /home/guangda/repos/labelmaker-mix3d
poetry run test --config-name="config_test_scannet20.yaml" general.checkpoint="$CKPT_PATH" data.test_mode=$TEST_MODE data.num_labels="20"
