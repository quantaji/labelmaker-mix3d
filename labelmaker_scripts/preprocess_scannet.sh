env_name=labelmaker-mix3d
eval "$(conda shell.bash hook)"
conda activate $env_name

DATA_DIR="data/raw/scannet/scannet"
SAVE_DIR="data/processed/scannet"
GIT_REPO="data/raw/scannet/ScanNet"

# preprocess
poetry run \
    python mix3d/datasets/preprocessing/scannet_preprocessing.py preprocess_sequential \
    --git_repo="$GIT_REPO" \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR" \
    --modes="(test,)" \
    --scannet200="true"
