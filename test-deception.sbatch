#!/bin/bash

#SBATCH -A exasgd
#SBATCH -p dl
#SBATCH -t 10
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o test.%J.out
#SBATCH -e test.%J.err


source ./buildsystem/deception-env.sh
./buildsystem/build.sh
cd build
ctest -VV
