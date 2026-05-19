#!/bin/bash
#PBS -N surface
#PBS -q full
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate agox_uma

cd $PBS_O_WORKDIR
python -u slab.py
exit 0

