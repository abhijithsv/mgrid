#!/bin/bash
# Job name:
#SBATCH --job-name=CAST

# use --exclusive to get the whole nodes exclusively for this job
#SBATCH --exclusive

#SBATCH --time=00:10:00

#SBATCH -p batch

# Set up work environment
#. /opt/intel/mkl/bin/mklvars.sh intel64
#
# Load your modules
module load intel/latest
module load mvapich2/2.0-icc-slurm
export MV2_ENABLE_AFFINITY=0
export MV2_DEFAULT_TIME_OUT=16
export MV2_DEFAULT_RETRY_COUNT=10
export MV2_DEFAULT_RNR_RETRY=10
#SBATCH --ntasks-per-node=8

EXE=grid_redistribute_test.x 
srun -N 32 -n 32 ./$EXE
#srun -N 80 -n 80 ./$EXE 2 2 2 16 2 2
#make
