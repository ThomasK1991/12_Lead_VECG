#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --job-name=vae_train
#SBATCH --output=/users/newc6477/VAE/12_Lead_VECG/LogOutputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

echo "🚀 Starting GPU Test Job..."

# -- Setup basic environment
source ~/.bashrc
module purge
module load cuda/11.8
echo "✅ Loaded CUDA module:"
module list

echo "🔍 Listing /usr/lib64/libcuda.so*:"
ls -l /usr/lib64/libcuda.so*

echo "🔍 Checking nvcc version:"
which nvcc
nvcc --version

# -- Activate Conda environment
CONDA_ENV="tf2.14"
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "🐍 Activated Conda env: $CONDA_ENV"
conda info --envs

# -- Set CUDA paths to match the loaded module (CUDA 11.8)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# Extend LD_LIBRARY_PATH to include CUDA 11.8 libraries and the conda environment's lib directory.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "🔧 Updated environment variables:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# -- Create symlink for libcuda.so in the conda environment if missing
if [ ! -f "$CONDA_PREFIX/lib/libcuda.so" ]; then
    echo "🔧 Creating symlink for libcuda.so in conda environment"
    ln -s /usr/lib64/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so
fi

# -- (Optional) Remove LD_PRELOAD as it may cause duplicate loading issues
# echo "LD_PRELOAD before unsetting: $LD_PRELOAD"
# unset LD_PRELOAD
# echo "LD_PRELOAD has been unset."

# -- Run tests
echo "🐍 Python executable: $(which python)"

echo -e "\n🖥️ Running nvidia-smi:"
nvidia-smi

echo -e "\n📦 TensorFlow GPU test:"
python -c "import tensorflow as tf; \
from tensorflow.python.platform import build_info as tf_build; \
print('TF Version:', tf.__version__); \
print('Built with CUDA:', tf_build.build_info.get('cuda_version', 'N/A')); \
print('Built with cuDNN:', tf_build.build_info.get('cudnn_version', 'N/A')); \
print('GPUs visible to TensorFlow:', tf.config.list_physical_devices('GPU'))"

echo "✅ Done."