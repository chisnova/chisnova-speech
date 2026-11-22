#!/bin/bash

rm -rf linux_build
mkdir linux_build
cd linux_build
cmake ../.
make -j4

