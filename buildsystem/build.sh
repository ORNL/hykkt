#!/bin/bash

mkdir -p build

rm -rf build/*

cmake -B build -S . && 

cmake --build build
