#!/bin/bash

#SBATCH --account=jessetho_1016
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

pip install torch
pip install torchvision
pip install -U scikit-learn scipy matplotlib
pip install wheel
pip install pandas

module purge
module load gcc/11.3.0
module load openblas/0.3.20
module load python
module load cuda/11.6.2
module load py-numpy/1.22.4
module load py-pillow/9.0.0

python3 Phase2_ResNet.py
