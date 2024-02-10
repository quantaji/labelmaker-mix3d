env_name=labelmaker-mix3d-preprocess
eval "$(conda shell.bash hook)"
conda activate $env_name

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

DATA_DIR="/media/hermann/data/labelmaker/ARKitScene_LabelMaker"
SAVE_DIR="data/processed/arkitscenes_labelmaker"

# preprocess
# python -m mix3d.datasets.preprocessing.arkitscenes_labelmaker preprocess \
#     --data_dir="$DATA_DIR" \
#     --save_dir="$SAVE_DIR" \
#     --n_jobs=8

python -m mix3d.datasets.preprocessing.arkitscenes_labelmaker preprocess_sequential \
    --data_dir="$DATA_DIR" \
    --save_dir="$SAVE_DIR"
