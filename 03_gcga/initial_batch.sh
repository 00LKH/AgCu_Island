#!/bin/bash
#PBS -N mace_mh
#PBS -q full
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -j oe

source /home/kyunghun/anaconda3/etc/profile.d/conda.sh
conda activate agox_uma
export PYTHONPATH=$PYTHONPATH:/home/kyunghun/08_GOCIA/gocia

cd $PBS_O_WORKDIR

python -u 00_prep_test_trajectory.py
python -u 01_initial_gcga_run.py
exit 0

