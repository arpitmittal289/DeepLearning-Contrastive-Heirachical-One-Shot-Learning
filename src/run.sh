#!/bin/bash
#SBATCH --acount=jessetho_1016

module purge
module load gcc/8.3.0
module load python/3.9.2

python3 baselineModels.py
