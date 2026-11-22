#!/bin/bash

echo "START LAYER TEST"

rm python/binary/*
rm -rf build
rm layer_tester

cd python
python3 main.py
cd ..

mkdir build
cd build
cmake ../.
make -j4
cp layer_tester ../.
cd ..
./layer_tester

rm -rf build
rm layer_tester
rm python/binary/*
