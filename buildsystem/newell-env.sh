
source /etc/profile.d/modules.sh
module purge

module use -a /qfs/projects/exasgd/src/cameron/spack/share/spack/modules/linux-centos8-power9le

module load gcc/8.5.0
module load cmake-3.23.2-gcc-8.5.0-tpplkft
module load cuda/11.4


