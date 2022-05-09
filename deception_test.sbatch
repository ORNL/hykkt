#!/bin/bash

#SBATCH -A exasgd
#SBATCH -p a100_shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:59:00
#SBATCH --gres=gpu:1
#SBATCH -o test.%J.out
#SBATCH -e test.%J.err


cd build
ctest -VV

