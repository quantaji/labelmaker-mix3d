env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

# CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_scannet200/2024-02-12_0629/18eacb2b/fd4360e3_10a2/epoch\=539-val_IoU_0\=0.269.ckpt" # new best performance of training scannet200 from scratch
# /home/guangda/repos/labelmaker-mix3d/saved/evaluation_scannet200/2024-02-22_1612/1b9abe1d/71750046_8d1c/metrics.csv

# CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_scannet200/2024-02-20_1458/1b9abe1d/21c8719e_c9a1/epoch\=279-val_IoU_0\=0.280.ckpt" # arkit_scenes pretrained ckpt, then fine tuned.
# /home/guangda/repos/labelmaker-mix3d/saved/evaluation_scannet200/2024-02-22_1636/1b9abe1d/3ff5dab6_bafb/metrics.csv

# CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_arkitscenes_scannet200/2024-02-18_2014/1b9abe1d/4cd47436_12e4/epoch\=199-val_IoU_0\=0.242.ckpt" # best joint train model
# /home/guangda/repos/labelmaker-mix3d/saved/evaluation_scannet200/2024-02-22_1638/1b9abe1d/b2dbd22b_5aca/metrics.csv

TEST_MODE="validation"
# TEST_MODE="test"
cd /home/guangda/repos/labelmaker-mix3d
poetry run test --config-name="config_test_scannet200.yaml" general.checkpoint="$CKPT_PATH" data.test_mode=$TEST_MODE
