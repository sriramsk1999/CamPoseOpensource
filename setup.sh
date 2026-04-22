#!/usr/bin/env bash
set -euo pipefail

# init submodules
git submodule update --init --recursive





CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n know_your_camera python=3.10 -y
conda activate know_your_camera

python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install numpy scipy h5py einops pillow tqdm imageio imageio-ffmpeg PyOpenGL glfw wandb pyyaml

pip install diffusers transformers

pip install mujoco 

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# only need for robosuite
pip install -e "$REPO_ROOT/robosuite_source"

# only need for maniskill
pip install -e "$REPO_ROOT/maniskill_source"


# download demos
pip install gdown
mkdir -p "$REPO_ROOT/temp"
gdown --folder --remaining-ok --id 1dmv-ueaP8F0ElqgVXsdmX-S9hvfQb7Yf -O "$REPO_ROOT/temp"
mkdir -p "$REPO_ROOT/policy_maniskill/demos" "$REPO_ROOT/policy_robosuite/demos"
mv -n "$REPO_ROOT/temp/demos_maniskill/"* "$REPO_ROOT/policy_maniskill/demos/"
mv -n "$REPO_ROOT/temp/demos_robosuite/"* "$REPO_ROOT/policy_robosuite/demos/"
rm -rf "$REPO_ROOT/temp"


echo "Environment 'know_your_camera' is ready."

