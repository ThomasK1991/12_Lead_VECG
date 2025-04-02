#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --output=/users/newc6477/VAE/12_Lead_VECG/LogOutputs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=70:00:00 
#Limit is 80hrs

echo "🚀 Starting VAE Training Job..."

# ===== Safe Bash Setup =====
abort() { >&2 printf '█%.0s' {1..40}; (>&2 printf "\n[ERROR] $(basename $0) has exited early\n"); exit 1; }
scriptdirpath=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
IFS=$'\n\t'; set -eo pipefail
trap 'abort' 0; set -u
pushd "${scriptdirpath}" > /dev/null

# ===== Load CUDA =====
echo "⚙️ Loading CUDA..."
module purge
module --ignore-cache load cuda/11.8


# ===== Activate Conda =====
CONDA_ENV="my_env"
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "🧠 Activating ${CONDA_ENV} env..."
  set +u; conda activate "${CONDA_ENV}"; set -u
fi



# ===== Set Up LD_LIBRARY_PATH Properly =====
# -- Set CUDA paths to match the loaded module (CUDA 11.8)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# Extend LD_LIBRARY_PATH to include CUDA 11.8 libraries and the conda environment's lib directory.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib_overrides:$LD_LIBRARY_PATH
# -- Create symlink for libcuda.so in the conda environment if missing
if [ ! -f "$CONDA_PREFIX/lib/libcuda.so" ]; then
    echo "🔧 Creating symlink for libcuda.so in conda environment"
    ln -s /usr/lib64/libcuda.so.1 $CONDA_PREFIX/lib/libcuda.so
fi

# ===== Debug Block =====
DEBUG=true
if [ "$DEBUG" == true ]; then
  echo "🐍 Python:"
  which python

  echo -e "\n🖥️ nvidia-smi:"
  nvidia-smi

  echo -e "\n📦 TensorFlow + GPU check:"
  python -c "import tensorflow as tf; print('GPUs visible to TensorFlow:', tf.config.list_physical_devices('GPU'))"
fi


# ===== Run Training =====
echo -e "\n🏃‍♂️ Running your VAE script...\n"
python /users/newc6477/VAE/12_Lead_VECG/src/main.py -p /users/newc6477/VAE/12_Lead_VECG/src/params.yml


# ===== Deactivate Conda and Wrap Up =====
conda deactivate
popd > /dev/null
trap : 0
(>&2 echo "✔ Job completed successfully.")
exit 0
