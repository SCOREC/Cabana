#!/bin/bash -ex 
cmake $1 \
-DCMAKE_BUILD_TYPE="Debug" \
-DCMAKE_CXX_COMPILER=nvcc_wrapper \
-DCabana_ENABLE_TESTING=ON \
-DCabana_ENABLE_EXAMPLES=ON \
-DCabana_REQUIRE_CUDA=ON \
-DCMAKE_INSTALL_PREFIX=$PWD/install