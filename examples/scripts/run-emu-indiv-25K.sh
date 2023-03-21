#!/bin/bash

export matrix_dir=/home/fda/matrices_blocks/25k
export matrix="ACTIVSg25k";

for i in {0..38}
#for i in {9..9}
do
    m1=$(printf "%02d" $i)
	echo "BEGIN-$m1"
    ./examples/solver_driver_emulator $matrix_dir "$matrix"_AC $m1 3 1000000
	echo "END-$m1"
done

