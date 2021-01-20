# CabanaM

CabanaM is an extension to [CoPA Cabana](https://github.com/ECP-copa/Cabana/wiki) to support grouping particles by the unstructured mesh element they reside in using a Compressed-Sparse Row setup.


## Build Instructions
* Create the working directory:
```
mkdir -p ~/develop/cabanam_kokkos3200
```

Kokkos and Cabana links:
[kokkos 3.2.0](https://github.com/kokkos/kokkos/releases/tag/3.2.00) ([build instructions](https://github.com/kokkos/kokkos/blob/master/BUILD.md))
and [cabana](https://github.com/ECP-copa/Cabana/blob/master/README.md) ([build instructions](https://github.com/ECP-copa/Cabana/wiki/Build-Instructions))

* Place the following commands into a script `~/develop/cabanam_kokkos3200/buildAll.sh`:
``` 
source ~/develop/cabanam_kokkos3200/envBlockade.sh

cd $root
git clone git@github.com:kokkos/kokkos.git
cd kokkos
git checkout 3.2.00
cd -
mkdir $kk #set in the environment script
cd $_
cmake ../kokkos \
  -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
  -DKokkos_ARCH_TURING75=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=off \
  -DKokkos_ENABLE_CUDA=on \
  -DKokkos_ENABLE_CUDA_LAMBDA=on \
  -DKokkos_ENABLE_DEBUG=on \
  -DKokkos_ENABLE_PROFILING=on \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24 install

cd $root
git clone git@github.com:SCOREC/Cabana.git cabana
cd cabana
git checkout cm_rebuild
cd -
mkdir build-cabana-blockade-cuda
cd $_
cmake ../cabana \
  -DCMAKE_BUILD_TYPE="Debug" \
  -DCMAKE_CXX_COMPILER=nvcc_wrapper \
  -DCabana_ENABLE_TESTING=ON \
  -DCabana_ENABLE_EXAMPLES=ON \
  -DCabana_REQUIRE_CUDA=ON \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j 24
sleep 3
ctest
```

* Make the script executable:
``` chmod +x ~/develop/cabanam_kokkos3200/buildAll.sh ```

* Run the script:
```
cd ~/develop/cabanam_kokkos3200/
./buildAll.sh
```

* Assuming the `buildAll.sh` script has been successfully run before, the following commands will allow you to rebuild Cabana after making some changes to the source:
```
cd ~/develop/cabanam_kokkos3200/
cd build-cabana-blockade-cuda
make
```