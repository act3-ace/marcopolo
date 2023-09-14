#!/bin/bash

# exit when any command fails
set -e

# script for setting up MarcoPolo conda environment
# use: . setup_MP.sh
#       The space is important, as it keeps it in the main shell
# Output is a conda environment called "marcopolo"

# Ubuntu installs first
apt-get install -y cmake python3-opengl

# Mamba install next
#  Check if mamba is installed, only install if it isn't already there
#   https://stackoverflow.com/questions/592620/how-can-i-check-if-a-program-exists-from-a-bash-script
#  Download
#  Batch install
#  Run init because batch install doesn't
#  Fix base environemt BS
#  Remove setup script
MAMBA_PATH=~/mambaforge/condabin
if ! command -v $MAMBA_PATH/mamba &> /dev/null
then
    # Install Mamba and clean up
    rm -rf /opt/conda
    wget -P ~ "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash ~/Mambaforge-$(uname)-$(uname -m).sh -b
    $MAMBA_PATH/mamba init
    $MAMBA_PATH/conda config --set auto_activate_base false
    rm ~/Mambaforge-$(uname)-$(uname -m).sh
fi


# Source for shell
#  This is required for create, activate, deactive below
source ~/mambaforge/etc/profile.d/conda.sh

# Create marcopolo env
#  Pin python version
#  Install required packages
#  "-y" installs without asking, same as bash
mamba create -n marcopolo -y python=3.9 gymnasium-box2d imageio matplotlib \
numpy networkx pandas ipykernel h5py pynng gifsicle lbzip2 black \
numba pre-commit pygame scipy tianshou packaging pytorch tensorflow \
-c conda -c conda-forge -c pytorch

# Pip installs
#  Because they aren't available in conda-forge
conda activate marcopolo
pip install --use-pep517 --no-cache-dir neat-python pettingzoo[classic]
./add_hooks.sh
conda deactivate

# reset failure handling
set +e

# should be good to go!
