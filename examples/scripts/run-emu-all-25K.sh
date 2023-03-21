#!/bin/bash

export matrix_dir=/home/fda/matrices_blocks/25k
export matrix="ACTIVSg25k";
export sequences="18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37";

./examples/solver_driver_emulator $matrix_dir "$matrix"_AC $sequences 3 1000000
