#ifndef CABANAM_HPP
#define CABANAM_HPP

#include <Cabana_MemberTypes.hpp>
#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Distributor.hpp>
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
      , _aosoa( NULL )
    {}

    CabanaM( const int *deg, const int elem_count ) {
      _numElms = elem_count;
      _vector_length = AoSoA_t::vector_length;
      fprintf(stderr, "0.001\n");
      fprintf(stderr, "elem_count: %d\n", elem_count);
      for (int i = 0; i < elem_count; i++) {
        fprintf(stderr, "deg[%d] = %d\n", i, deg[i]);
      }
      _offsets = buildOffset(deg,elem_count);
      _numSoa = _offsets[elem_count];
      _capacity = _numSoa*_vector_length;
      fprintf(stderr, "0.002\n");
      assert( cudaSuccess == cudaDeviceSynchronize());
      _aosoa = makeAoSoA(_capacity,_numSoa);
      assert( cudaSuccess == cudaDeviceSynchronize());
      _parentElm = getParentElms(_numElms,_numSoa,_offsets);
      setActive(_aosoa, _numSoa, deg, _parentElm, _offsets);
    }

    AoSoA_t* makeAoSoA(const int capacity, const int numSoa) {
      auto aosoa = new AoSoA_t();
      aosoa->resize(capacity);
      assert(numSoa == aosoa->numSoA());
      return aosoa;
    }

    int* buildOffset(const int* deg, const int elem_count) {
      auto offset = new int[elem_count+1];
      // elem at i owns SoA offsets[i+1] - offsets[i]
      offset[0] = 0;
      for ( int i=0; i<elem_count; ++i ) {
        const auto SoA_count = (deg[i]/_vector_length) + 1;
        offset[i+1] = SoA_count + offset[i];
      }
      return offset;
    }

    int* getParentElms(const int numElms,const int numSoa, const int* offsets) {
      auto elms = new int[numSoa];
      for( int elm=0; elm<numElms; elm++ )
        for( int soa=offsets[elm]; soa<offsets[elm+1]; soa++)
          elms[soa]=elm;
      return elms;
    }

    void setActive(AoSoA_t* aosoa, const int numSoa, const int* deg, 
        const int* parent, const int* offsets) {
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

      const auto soaLen = _vector_length;
      const auto cap = capacity();
      const auto activeSliceIdx = aosoa->number_of_members-1;
      printf("number of member types %d\n", activeSliceIdx+1);
      auto active = slice<activeSliceIdx>(*aosoa);
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
    AoSoA_t* aosoa() { return _aosoa; }

    KOKKOS_FUNCTION
    AoSoA_t& aosoaRef() { return *_aosoa; }

    void rebuild() {
      const auto soaLen = AoSoA_t::vector_length;
      Kokkos::View<int*> elmDegree("elmDegree", _numElms);
      Kokkos::View<int*> elmOffsets("elmOffsets", _numElms);
      auto newParent = slice<0>(*aosoa());
      const auto activeSliceIdx = _aosoa->number_of_members-1;
      printf("number of member types %d\n", activeSliceIdx+1);
      auto active = slice<activeSliceIdx>(*aosoa());
      auto elmDegree_d = Kokkos::create_mirror_view_and_copy(memspace(), elmDegree);
      assert( cudaSuccess == cudaDeviceSynchronize());
      //first loop to count number of particles per new element (atomic)
      auto atomic = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          auto parent = newParent.access(soa,tuple);
          Kokkos::atomic_increment<int>(&elmDegree_d(parent));
        }
      };
      assert( cudaSuccess == cudaDeviceSynchronize());
      Cabana::SimdPolicy<soaLen,exespace> simd_policy( 0, capacity() );
      Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
      auto elmDegree_h = Kokkos::create_mirror_view_and_copy(hostspace(), elmDegree_d);
      //prepare a new aosoa to store the shuffled particles
      auto newOffset = buildOffset(elmDegree_h.data(), _numElms);
      const auto newNumSoa = newOffset[_numElms];
      const auto newCapacity = newNumSoa*_vector_length;
      auto newAosoa = makeAoSoA(newCapacity, newNumSoa);
      printf("newCapacity: %d, newNumSoa: %d\n", newCapacity, newNumSoa);
      //assign the particles from the current aosoa to the newAosoa 
      Kokkos::View<int*,hostspace> newOffset_h("newOffset_host",_numElms+1);
      for (int i=0; i<=_numElms; i++)
        newOffset_h(i) = newOffset[i];
      assert( cudaSuccess == cudaDeviceSynchronize());
      auto newOffset_d = Kokkos::create_mirror_view_and_copy(memspace(), newOffset_h);
      Kokkos::View<int*, hostspace> elmPtclCounter_h("elmPtclCounter_device",_numElms); 
      auto elmPtclCounter_d = Kokkos::create_mirror_view_and_copy(memspace(), elmPtclCounter_h);
      auto newActive = slice<activeSliceIdx>(*newAosoa);
      assert( cudaSuccess == cudaDeviceSynchronize());
      setvbuf( stdout, NULL, _IONBF, 0 );


      auto aosoaTest = new AoSoA_t();
      aosoaTest->resize(32);
      auto tupleTest = new Tuple<Cabana::MemberTypes<int,int>>();      
      auto copyTest = KOKKOS_LAMBDA(const int& i) {
        aosoaTest->setTuple(i, *tupleTest);
      };
      Kokkos::parallel_for("copyTest", Kokkos::RangePolicy<exespace>(0, 32), copyTest);
      
      Kokkos::fence();

      assert( cudaSuccess == cudaDeviceSynchronize());

      
      printf("_vector_length: %d\n", _vector_length);
      auto _vector_lengthTEST = _vector_length;
      auto aosoa_TEST = _aosoa;

      auto copyPtcls = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          //Compute the destSoa based on the destParent and an array of
          // counters for each destParent tracking which particle is the next
          // free position. Use atomic fetch and incriment with the
          // 'elmPtclCounter_d' array.

          //Tuple<DataTypes> apple = Tuple<DataTypes>(oldTuple);
          //Tuple<Cabana::MemberTypes<int,int>> apple = Tuple<Cabana::MemberTypes<int,int>>();


          //std::integral_constant<std::size_t, 10> orange = std::integral_constant<std::size_t, 10>();
          //printf("%d\n",orange());

          // WHAT IS HAPPENING HERE
          //printf("soa: %d, _vector_length: %d, tuple: %d\n", soa, _vector_length, tuple);
          printf("soa: %d, _vector_lengthTEST: %d, tuple: %d\n", soa, _vector_lengthTEST, tuple);
          printf("0.0001\n");
          auto destParent = newParent.access(soa,tuple); // trying to access slice<0>(*aosoa())
          printf("0.0002\n");
          auto occupiedTuples = Kokkos::atomic_fetch_add<int>(&elmPtclCounter_d(destParent), 1);
          printf("0.0003\n");
          printf("soa_c: %d, tuple_c: %d\n", soa, tuple);
          printf("added stuff: %d\n", soa * _vector_lengthTEST + tuple); // correct
          //auto oldTuple = _aosoa->getTuple(soa * _vector_length + tuple);
          auto oldTuple = aosoa_TEST->getTuple(soa * _vector_lengthTEST + tuple);
          auto testTuple = oldTuple;
          
          //aosoa_TEST->setTuple(soa * _vector_lengthTEST + tuple, apple); // TEST

          printf("0.0004\n");
          auto firstSoa = newOffset_d(destParent);
          printf("0.0005\n");
          // use newOffset_d to figure out which soa is the first for destParent
          //newAosoa->setTuple(firstSoa * _vector_length + occupiedTuples, oldTuple);
          newAosoa->setTuple(firstSoa * _vector_lengthTEST + occupiedTuples, oldTuple);

          printf("0.0006\n");
          printf("active particle which was at soa %d and tuple %d has been moved to soa %d and tuple %d\n", soa, tuple, firstSoa, occupiedTuples); 
        }
      };
      Cabana::simd_parallel_for(simd_policy, copyPtcls, "copyPtcls");
      //Kokkos::parallel_for("copyPtcls", Kokkos::MDRangePolicy< Kokkos::Rank<2> >( {0,0}, {newNumSoa, soaLen} ), copyPtcls);
      
      assert( cudaSuccess == cudaDeviceSynchronize());
      cudaDeviceSynchronize();
      //destroy the old aosoa and use the new one in the CabanaM object
      delete _aosoa;
      assert( cudaSuccess == cudaDeviceSynchronize());
      _aosoa = newAosoa;
      assert( cudaSuccess == cudaDeviceSynchronize());
    }

  private:
    std::size_t _capacity;
    std::size_t _numElms;
    std::size_t _numSoa;
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    int *_parentElm; // parent elm for each soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP

