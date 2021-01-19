# CabanaM

CabanaM is an extension to [CoPA Cabana](https://github.com/ECP-copa/Cabana/wiki) to support grouping particles by the unstructured mesh element they reside in using a Compressed-Sparse Row setup.


## Added Functions

`CabanaM( const int *deg, const int elem_count )`
##### Parameters:
* deg: pointer to an array of ints, representing the number of active elements in each SoA
* elem_count: length of 'deg'
##### Purpose:
Constructor

`AoSoA_t makeAoSoA( const int capacity, const int numSoa )`
##### Parameters:
* capacity: capacity of the AoSoA to be created
* numSoa: total number of SoAs in AoSoA (including active mask)
##### Purpose:
Initialize an AoSoA of capacity 'capacity' spread across a number of SoAs 'numSoa'

`int* buildOffset( const int* deg, const int elem_count )`
##### Parameters:
* deg: pointer to an array of ints, representing the number of active elements in each SoA
* elem_count: length of 'deg'
##### Purpose:
Builds the offset array for the CSR.

`int* getParentElms( const int numElms, const int numSoa, const int* offsets )`
##### Parameters:
* numElms: total number of element SoAs in AoSoA
* numSoa: total number of SoAs in AoSoA (including active mask)
* offsets: offset array for AoSoA, built by 'buildOffset'
##### Purpose:
Builds the parent array for tracking particle position, where each element is an int
representing the SoA each particle resides in.

`void setActive ( AoSoA_t &aosoa, const int numSoa, const int* deg, const int* parent, const int* offsets )`
##### Parameters:
* aosoa: the AoSoA to be edited
* numSoa: total number of SoAs in AoSoA (including active mask)
* deg: pointer to an array of ints, representing the number of active elements in each SoA
* parent: parent array for AoSoA, built by 'getParentElms'
* offsets: offset array for AoSoA, built by 'buildOffset'
##### Purpose:
Fill/Refill last SoA in AoSoA with a series of 1s and 0s where 1 denotes an active particle and 0 denotes an inactive particle.

`void rebuild( Kokkos::View<int*,MemorySpace> newParent )`
##### Parameters:
* newParent: A Kokkos View of ints in the same memory space as this CabanaM instance,
representing the new SoAs of each individual element in the AoSoA
##### Purpose:
Fully rebuild the AoSoA with these new parent SoAs by copying into a new AoSoA and overwriting the old one.


## Build Instructions
* Create the working directory:
```
mkdir -p ~/develop/cabanam_kokkos3200
```

* Dependency Stuff (Is this stuff specific to Blockade?? How do I expand it out??????)

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
source envBlockade.sh
cd build-cabana-blockade-cuda
make
```
Note, `source envBlockade.sh` only needs to be run once for each terminal/session to setup the environment.