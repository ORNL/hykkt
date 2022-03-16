#!/bin/sh
#SBATCH -A exasgd
#SBATCH -p a100_shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:59:00
#SBATCH --gres=gpu:1
module load cmake/3.15.3
module load gcc/7.5.0
module load cuda/11.1
#for i in 0{0..9} {10..48};
for i in 0{9..9};
do
    echo
      echo "reading matrix: $i";
        #  echo "reading H"
          #hfile="/people/rege393/Hybrid_dev/mats/a2000/block_H_matrix_ACTIVSg2000_AC_$i.mtx"
          hfile="mats/a2000/block_H_matrix_ACTIVSg2000_AC_$i.mtx"
        #  echo "reading Ds"
          Dsfile="mats/a2000/block_Dd_matrix_ACTIVSg2000_AC_$i.mtx"
      #  echo "reading jc"
          jcfile="mats/a2000/block_J_matrix_ACTIVSg2000_AC_$i.mtx"
      #  echo "reading jd"
          jdfile="mats/a2000/block_Jd_matrix_ACTIVSg2000_AC_$i.mtx"
       #   echo "reading rx"
          rxfile="mats/a2000/block_rx_ACTIVSg2000_AC_$i.mtx"
        #  echo "reading rs"
          rsfile="mats/a2000/block_rs_ACTIVSg2000_AC_$i.mtx"
         # echo "reading ry"
          ryfile="mats/a2000/block_ry_ACTIVSg2000_AC_$i.mtx"
        #  echo "reading ryd"
          rydfile="mats/a2000/block_ryd_ACTIVSg2000_AC_$i.mtx"
        #  echo "reading permutation"
       #   permfile="/people/rege393/Hybrid_dev/perm_test/permutation2000.mtx"
       #   echo "reading permuted and scaled matrices"
#          permJtfile="/people/rege393/Hybrid_dev/perm_test/perm_Jt_ACTIVSg200_AC_09.mtx"
 #         permJfile="/people/rege393/Hybrid_dev/perm_test/perm_J_ACTIVSg200_AC_09.mtx"
  #        permHfile="/people/rege393/Hybrid_dev/perm_test/perm_H_ACTIVSg200_AC_09.mtx"
         
        srun ./hybrid_solver $hfile $Dsfile $jcfile $jdfile $rxfile $rsfile $ryfile $rydfile 3 10000.0 #$permfile 
        #$permJfile $permJtfile $permHfile
      done 




