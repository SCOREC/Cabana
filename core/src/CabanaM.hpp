#ifndef CABANAM_HPP
#define CABANAM_HPP

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <cassert>

namespace Cabana
{

template<class DataTypes,
         class MemorySpace>
class CabanaM
{
  public:

    using CM_DT = Cabana::AppendMT<int,DataTypes>;
    using AoSoA_t = Cabana::AoSoA<typename CM_DT::type,MemorySpace>;
    using memspace = typename MemorySpace::memory_space;
    using exespace = typename MemorySpace::execution_space;
    using hostspace = Kokkos::HostSpace;

  public:
    CabanaM()
      : _capacity( 0 )
      , _numElms( 0 )
      , _numSoa( 0 )
      , _vector_length( 0 )
      , _offsets( NULL )
      , _parentElm( NULL )
    {}

    /**
     * Constructor
     * @param[in] deg pointer to an array of ints, representing the number of active particles in each SoA
     * @param[in] elem_count length of deg
    */
    CabanaM( const int *deg, const int elem_count ) {
      _numElms = elem_count;
      _vector_length = AoSoA_t::vector_length;
      _offsets = buildOffset(deg,elem_count);
      _numSoa = _offsets[elem_count];
      _capacity = _numSoa*_vector_length;
      _aosoa = makeAoSoA(_capacity,_numSoa);
      _parentElm = getParentElms(_numElms,_numSoa,_offsets);
      setActive(_aosoa, _numSoa, deg, _parentElm, _offsets);
    }

    /**
     * Initialize an AoSoA (including hidden active SoA)
     * @param[in] capacity maximum capacity (number of particles) of the AoSoA to be created
     * @param[in] numSoa total number of SoAs (can be greater than elem_count if
     * any element of deg is _vector_length)
     * @return AoSoA of max capacity, capacity, and total number of SoAs, numSoa
    */
    AoSoA_t makeAoSoA( const int capacity, const int numSoa ) {
      auto aosoa = AoSoA_t();
      aosoa.resize(capacity);
      assert(numSoa == aosoa.numSoA());
      return aosoa;
    }

    /**
     * Builds the offset array for the CSR
     * @param[in] deg pointer to an array of ints, representing the number of active elements in each SoA
     * @param[in] elem_count length of deg
     * @return offset array (each element is the first index of each SoA block)
    */
    int* buildOffset( const int* deg, const int elem_count ) {
      auto offset = new int[elem_count+1];
      // elem at i owns SoA offsets[i+1] - offsets[i]
      offset[0] = 0;
      for ( int i=0; i<elem_count; ++i ) {
        const auto SoA_count = (deg[i]/_vector_length) + 1;
        offset[i+1] = SoA_count + offset[i];
      }
      return offset;
    }

    /**
     * Builds the parent array for tracking particle position
     * @param[in] numElms total number of element SoAs in AoSoA
     * @param[in] numSoa total number of SoAs (can be greater than elem_count if
     * any element of deg is _vector_length)
     * @param[in] offsets offset array for AoSoA, built by buildOffset
     * @return parent array, each element is an int representing the SoA each particle resides in
    */
    int* getParentElms( const int numElms, const int numSoa, const int* offsets ) {
      auto elms = new int[numSoa];
      for( int elm=0; elm<numElms; elm++ )
        for( int soa=offsets[elm]; soa<offsets[elm+1]; soa++)
          elms[soa]=elm;
      return elms;
    }

    /**
     * Fill/Refill last SoA in AoSoA with a series of 1s and 0s
     * where 1 denotes an active particle and 0 denotes an inactive particle.
     * @param[out] aosoa the AoSoA to be edited
     * @param[in] numSoa total number of SoAs (can be greater than elem_count if
     * any element of deg is _vector_length)
     * @param[in] deg pointer to an array of ints, representing the number of active elements in each SoA
     * @param[in] parent parent array for AoSoA, built by getParentElms
     * @param[in] offsets offset array for AoSoA, built by buildOffset
    */
    void setActive( AoSoA_t &aosoa, const int numSoa, const int* deg, 
        const int* parent, const int* offsets ) {
      Kokkos::View<int*,hostspace> deg_h("degree_host",_numElms);
      for (int i=0; i<_numElms; i++)
        deg_h(i) = deg[i];
      auto deg_d = Kokkos::create_mirror_view_and_copy(
          memspace(), deg_h);

      Kokkos::View<int*,hostspace> parent_h("parent_host",numSoa);
      for (int i=0; i<numSoa; i++)
        parent_h(i) = parent[i];
      auto parent_d = Kokkos::create_mirror_view_and_copy(memspace(), parent_h);

      Kokkos::View<int*,hostspace> offset_h("offset_host",_numElms+1);
      for (int i=0; i<=_numElms; i++)
        offset_h(i) = offsets[i];
      auto offset_d = Kokkos::create_mirror_view_and_copy(memspace(), offset_h);

      const int soaLen = _vector_length;
      const auto cap = capacity();
      const auto activeSliceIdx = aosoa.number_of_members-1;
      printf("number of member types %d\n", activeSliceIdx+1);
      auto active = slice<activeSliceIdx>(aosoa);
      Cabana::SimdPolicy<AoSoA_t::vector_length,exespace> simd_policy( 0, cap );
      Cabana::simd_parallel_for(simd_policy,
        KOKKOS_LAMBDA( const int soa, const int ptcl ) {
          const auto elm = parent_d(soa);
          const auto numsoa = offset_d(elm+1)-offset_d(elm);
          const auto lastsoa = offset_d(elm+1)-1;
          const auto elmdeg = deg_d(elm);
          const auto lastSoaDeg = soaLen - ((numsoa * soaLen) - elmdeg);
          int isActive = 0;
          if( soa < lastsoa ) {
            isActive = 1;
          }
          if( soa == lastsoa && ptcl < lastSoaDeg ) {
            isActive = 1;
          }
          printf("elm %3d deg %3d soa %3d lastsoa %3d lastSoaDeg %3d ptcl %3d active %1d\n",
            elm, elmdeg, soa, lastsoa, lastSoaDeg, ptcl, isActive);
          active.access(soa,ptcl) = isActive;
      }, "set_active");
    }

    KOKKOS_FUNCTION
    std::size_t numSoa() const { return _numSoa; }

    KOKKOS_FUNCTION
    int capacity() const { return _capacity; };

    KOKKOS_FUNCTION
    std::size_t numElements() const { return _numElms; }

    KOKKOS_FUNCTION
    std::size_t vector_length() const { return _vector_length; }

    KOKKOS_FUNCTION
    int offset(int i) const { return _offsets[i]; }

    KOKKOS_FUNCTION
    int parentElm(int i) const { return _parentElm[i]; }

    KOKKOS_FUNCTION
    AoSoA_t aosoa() { return _aosoa; }

    /**
     * Fully rebuild the AoSoA with these new parent SoAs by copying into a new AoSoA and overwriting the old one.
     * @param[in] newParent A Kokkos View of ints in the same memory space as this CabanaM instance,
     * representing the new SoAs of each individual element in the AoSoA
    */
    void rebuild( Kokkos::View<int*,MemorySpace> newParent ) {
      // Note: Use "assert( cudaSuccess == cudaDeviceSynchronize() );" to check for GPU issues
      const auto soaLen = AoSoA_t::vector_length;
      Kokkos::View<int*> elmDegree("elmDegree", _numElms);
      Kokkos::View<int*> elmOffsets("elmOffsets", _numElms);
      const auto activeSliceIdx = _aosoa.number_of_members-1;
      printf("number of member types %d\n", activeSliceIdx+1);
      auto active = slice<activeSliceIdx>(aosoa());
      auto elmDegree_d = Kokkos::create_mirror_view_and_copy(memspace(), elmDegree);
      //first loop to count number of particles per new element (atomic)
      auto atomic = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          auto parent = newParent((soa*soaLen)+tuple);
          Kokkos::atomic_increment<int>(&elmDegree_d(parent));
        }
      };
      Cabana::SimdPolicy<soaLen,exespace> simd_policy( 0, capacity() );
      Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
      auto elmDegree_h = Kokkos::create_mirror_view_and_copy(hostspace(), elmDegree_d);
      
      //prepare a new aosoa to store the shuffled particles
      auto newOffset = buildOffset(elmDegree_h.data(), _numElms);
      const auto newNumSoa = newOffset[_numElms];
      const auto newCapacity = newNumSoa*_vector_length;
      auto newAosoa = makeAoSoA(newCapacity, newNumSoa);
      //assign the particles from the current aosoa to the newAosoa 
      Kokkos::View<int*,hostspace> newOffset_h("newOffset_host",_numElms+1);
      for (int i=0; i<=_numElms; i++)
        newOffset_h(i) = newOffset[i];
      auto newOffset_d = Kokkos::create_mirror_view_and_copy(memspace(), newOffset_h);
      Kokkos::View<int*, hostspace> elmPtclCounter_h("elmPtclCounter_device",_numElms); 
      auto elmPtclCounter_d = Kokkos::create_mirror_view_and_copy(memspace(), elmPtclCounter_h);
      auto newActive = slice<activeSliceIdx>(newAosoa);
      auto aosoa_cp = _aosoa; // copy of member variable _aosoa (Kokkos doesn't like member variables)
      auto copyPtcls = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          //Compute the destSoa based on the destParent and an array of
          // counters for each destParent tracking which particle is the next
          // free position. Use atomic fetch and incriment with the
          // 'elmPtclCounter_d' array.
          auto destParent = newParent(soa*soaLen + tuple);
          auto occupiedTuples = Kokkos::atomic_fetch_add<int>(&elmPtclCounter_d(destParent), 1);
          auto oldTuple = aosoa_cp.getTuple(soa*soaLen + tuple);
          auto firstSoa = newOffset_d(destParent);
          // use newOffset_d to figure out which soa is the first for destParent
          newAosoa.setTuple(firstSoa*soaLen + occupiedTuples, oldTuple);
          printf("active particle which was at soa %d and tuple %d has been moved to soa %d and tuple %d\n", soa, tuple, firstSoa, occupiedTuples); 
        }
      };
      Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");
      //destroy the old aosoa and use the new one in the CabanaM object
      _aosoa = newAosoa;
      setActive(_aosoa, _numSoa, elmDegree_h.data(), _parentElm, _offsets);
    }

  private:
    std::size_t _capacity;
    std::size_t _numElms;
    std::size_t _numSoa;
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    int *_parentElm; // parent elm for each soa
    AoSoA_t _aosoa;

};

} // end namespace Cabana

#endif // CABANAM_HPP

