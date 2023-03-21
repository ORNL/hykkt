#!/bin/bash

export matrix_dir=/home/fda/matrices_blocks/70k
export matrix="ACTIVSg70k";
export sequences="01,02,03,04,05,06,07,08,09";

./examples/solver_driver_emulator $matrix_dir "$matrix"_AC $sequences 3 1000000
