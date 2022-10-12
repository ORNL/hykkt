#!/bin/bash

#SBATCH -A exasgd
#SBATCH -p a100_shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:59:00
#SBATCH --gres=gpu:1
#SBATCH -o test.%J.out
#SBATCH -e test.%J.err

set -x

# Load modules
source /etc/profile.d/modules.sh

module purge 
module load cmake/3.19.6
module load gcc/9.1.0
module load cuda/11.4

ulimit


mkdir -p build

rm -rf build/*

cmake -B build -S . && 

cmake --build build


cd build
ctest -VV

EXIT_CODE=$?
echo BUILD_STATUS:$EXIT_CODE
exit $EXIT_CODE


