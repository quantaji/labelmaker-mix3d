env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH

# CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_arkitscenes_scannet200/2024-02-18_2014/1b9abe1d/4cd47436_12e4/epoch\=199-val_IoU_0\=0.242.ckpt" # best joint train model
# /home/guangda/repos/labelmaker-mix3d/saved/evaluation_arkit/2024-02-22_1959/1b9abe1d/264ba713_1a18/hparams.yaml

# CKPT_PATH="/home/guangda/repos/labelmaker-mix3d/saved/baseline_arkitscenes/2024-02-19_1444/1b9abe1d/20419e2c_5d34/epoch\=439-val_IoU_0\=0.232.ckpt" # best pre train model
# /home/guangda/repos/labelmaker-mix3d/saved/evaluation_arkit/2024-02-22_1957/1b9abe1d/6f4e8bb7_39cf/metrics.csv

TEST_MODE="validation"
# TEST_MODE="test"

cd /home/guangda/repos/labelmaker-mix3d
poetry run test --config-name="config_test_arkit.yaml" general.checkpoint="$CKPT_PATH" data.test_mode=$TEST_MODE data.batch_size=2
