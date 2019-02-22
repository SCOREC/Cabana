set -ex 
cmake $1 \
-DCMAKE_BUILD_TYPE="Debug" \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCabana_ENABLE_TESTING=ON \
-DCabana_ENABLE_EXAMPLES=ON \
-DCabana_ENABLE_Serial=ON \
-DCabana_ENABLE_OpenMP=ON \
-DCabana_ENABLE_Cuda:BOOL=ON \
-DCabana_ENABLE_MPI=ON \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
