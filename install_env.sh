set -e

env_name=labelmaker-mix3d
conda create --name $env_name --yes python=3.9
eval "$(conda shell.bash hook)"
conda activate $env_name

INSTALLED_GCC_VERSION="8.5.0"
INSTALLED_CUDA_VERSION="11.8.0"
INSTALLED_CUDA_ABBREV="cu118"
INSTALLED_PYTORCH_VERSION="2.0.0"
INSTALLED_TORCHVISION_VERSION="0.15.1"

conda install -y -c conda-forge gxx=${INSTALLED_GCC_VERSION}
conda install -y -c "nvidia/label/cuda-${INSTALLED_CUDA_VERSION}" cuda
conda install -y -c anaconda openblas="0.3.20"

conda deactivate
conda activate ${env_name}

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export FORCE_CUDA=1
export MAX_JOBS=6
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"

# install torch
pip install torch==${INSTALLED_PYTORCH_VERSION} torchvision==${INSTALLED_TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${INSTALLED_CUDA_ABBREV}

export ENV_FOLDER="$(pwd)/$(dirname "$0")"
cd ${ENV_FOLDER}
rm -rf MinkowskiEngine
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

pip install -f --upgrade poetry dvc
poetry install

# install torch
pip install torch==${INSTALLED_PYTORCH_VERSION} torchvision==${INSTALLED_TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${INSTALLED_CUDA_ABBREV}

# neptune
pip install neptune

# open3d
pip install https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp39-cp39-manylinux_2_17_x86_64.whl
