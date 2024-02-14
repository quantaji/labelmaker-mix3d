env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

cd /home/guangda/repos/labelmaker-mix3d
poetry run test --config-name="config_test_scannet200.yaml" general.checkpoint="/home/guangda/repos/labelmaker-mix3d/saved/baseline_scannet200/2024-02-12_0629/18eacb2b/fd4360e3_10a2/epoch\=539-val_IoU_0\=0.269.ckpt" data.test_mode="test"

# poetry run test --config-name="config_test_scannet200.yaml" general.checkpoint="/home/guangda/repos/labelmaker-mix3d/saved/baseline_scannet200/2024-02-13_0258/18eacb2b/ec19bec5_c588/epoch\=419-val_IoU_0\=0.263.ckpt" data.test_mode="test"

