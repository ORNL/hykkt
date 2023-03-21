#!/bin/bash

export matrix_dir=/home/fda/matrices_blocks/70k/
export matrix="ACTIVSg70k"

for i in {0..15}
#for i in {9..9}
do
    m1=$(printf "%02d" $i)
	echo "BEGIN-$m1"
    ./examples/solver_driver_emulator $matrix_dir "$matrix"_AC $m1 3 1000000
	echo "END-$m1"
done

