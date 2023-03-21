#!/bin/bash

export matrix_dir=/home/fda/matrices_blocks/10k
export matrix="ACTIVSg10k";
export sequences="14,15,16,17,18,19,20";

./examples/solver_driver_emulator $matrix_dir "$matrix"_AC $sequences 3 1000000
