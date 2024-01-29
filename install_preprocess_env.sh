set -e

env_name=labelmaker-mix3d-preprocess
conda create --name $env_name --yes python=3.8
eval "$(conda shell.bash hook)"
conda activate $env_name

conda deactivate
conda activate ${env_name}

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

conda deactivate
conda activate ${env_name}

# tune
pip install scikit-learn pandas fire==0.2.1 loguru==0.4.1 PyYAML tqdm plyfile==0.7.1 natsort==7.0.1

# open3d
pip install "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp38-cp38-manylinux_2_17_x86_64.whl"
# install labelmaker
pip install -U "git+https://github.com/cvg/LabelMaker.git"
