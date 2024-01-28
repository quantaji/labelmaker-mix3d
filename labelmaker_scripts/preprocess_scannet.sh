env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export ENV_FOLDER="$(pwd)/$(dirname "$0")"
export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export PATH="$conda_home/bin:$PATH"
export CUDA_HOME=$CUDA_PATH
export FORCE_CUDA=1
export MAX_JOBS=6
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"

DATA_DIR="data/raw/scannet/scannet"
SAVE_DIR="data/processed/scannet200"
GIT_REPO="data/raw/scannet/ScanNet"

# preprocess
python -m mix3d.datasets.preprocessing.scannet_preprocessing preprocess_sequential \
    --git_repo="$GIT_REPO" \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR" \
    --scannet200="true"
