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
* Do Thing
* Do Thing
* Do Another Thing