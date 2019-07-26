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

  public:
    CabanaM()
      : _numPtcls( 0 )
      , _numElms( 0 )
      , _numSoa( 0 )
      , _vector_length( 0 )
      , _offsets( NULL )
      , _parentElm( NULL )
      , _aosoa( NULL )
    {}

    CabanaM( const int *deg, const int elem_count )
    {
      _aosoa = new AoSoA_t();
      //int vector_len = _aosoa->vector_length;
      _vector_length = Impl::PerformanceTraits<
             typename MemorySpace::execution_space>::vector_length;
      _offsets = new int[elem_count+1]();
      _numElms = elem_count;
      // elem at i owns SoA offsets[i+1] - offsets[i]
      _offsets[0] = 0;
      for ( int i=0; i<elem_count; ++i ) {
        int SoA_count = (deg[i]/_vector_length) + 1;
        _offsets[i+1] = SoA_count + _offsets[i];
        printf("%3d soa_count %3d\n", i, SoA_count);
      }
      _numSoa = _offsets[elem_count];
      _numPtcls = _offsets[elem_count]*_vector_length;
      _aosoa->resize(_numPtcls);
      assert(_numSoa == _aosoa->numSoA());
      _parentElm = new int[_numSoa];
      for( int elm=0; elm<elem_count; elm++ )
        for( int soa=_offsets[elm]; soa<_offsets[elm+1]; soa++)
          _parentElm[soa]=elm;
      setActive(deg);
    }

    void setActive(const int* deg) {
      Kokkos::View<int*,Kokkos::HostSpace> deg_h("degree_host",_numElms);
      for (int i=0; i<_numElms; i++)
        deg_h(i) = deg[i];
      auto deg_d = Kokkos::create_mirror_view_and_copy(
          memspace(), deg_h);

      Kokkos::View<int*,Kokkos::HostSpace> parent_h("parent_host",_numSoa);
      for (int i=0; i<_numSoa; i++)
        parent_h(i) = _parentElm[i];
      auto parent_d = Kokkos::create_mirror_view_and_copy(memspace(), parent_h);

      Kokkos::View<int*,Kokkos::HostSpace> offset_h("offset_host",_numElms+1);
      for (int i=0; i<=_numElms; i++)
        offset_h(i) = _offsets[i];
      auto offset_d = Kokkos::create_mirror_view_and_copy(memspace(), offset_h);

      const auto cap = capacity();
      const auto activeSliceIdx = _aosoa->number_of_members-1;
      printf("number of member types %d\n", activeSliceIdx+1);
      auto active = slice<activeSliceIdx>(*_aosoa);
      Cabana::SimdPolicy<AoSoA_t::vector_length,Kokkos::Cuda> simd_policy( 0, cap );
      Cabana::simd_parallel_for(simd_policy,
        KOKKOS_LAMBDA( const int soa, const int ptcl ) {
          const auto elm = parent_d(soa);
          const auto numsoa = offset_d(elm+1)-offset_d(elm);
          const auto lastsoa = offset_d(elm+1)-1;
          const auto elmdeg = deg_d(elm);
          const auto soaLen = 32; //FIXME
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
    std::size_t numParticles() const { return _numPtcls; }

    KOKKOS_FUNCTION
    int capacity() const { return _vector_length * _aosoa->numSoA(); }

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

   void rebuild() {
      const int SIMD_WIDTH = 32; //FIXME
      int sizesArray[_numElms];
      Kokkos::View<int*> elmDegree("elmDegree", _numElms);
      Kokkos::View<int*> elmOffsets("elmOffsets", _numElms);
      auto newParent = slice<0>(*aosoa());
      auto active = slice<1>(*aosoa());
      using ExecutionSpace = Kokkos::DefaultExecutionSpace;
      //first loop to count number of particles per new element (atomic)
      auto atomic = KOKKOS_LAMBDA(const int& soa,const int& tuple){
        if (active.access(soa,tuple) == 1){
          auto parent = newParent.access(soa,tuple);
          Kokkos::atomic_increment<int>(&elmDegree(parent));
        }
      };
      Cabana::SimdPolicy<SIMD_WIDTH,ExecutionSpace> simd_policy( 0, capacity() );
      Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );

      //print the number of particles per new element
      Kokkos::parallel_for(_numElms, KOKKOS_LAMBDA(const int i){
         printf("Degree of element %d = %d\n", i, elmDegree(i));
      });

      //build the offsets array - is this necessary?
      Kokkos::parallel_scan (_numElms, KOKKOS_LAMBDA (const int& i, int& upd, const bool &last) {
          const int val_i = elmDegree(i);
          if (last){
            elmOffsets(i) = upd;
          }
          upd+= val_i;
      });

      Kokkos::parallel_for(_numElms, KOKKOS_LAMBDA(const int i){
        auto current_offset = elmOffsets(i);
        printf("Offset of %d at position %d\n", current_offset, i);
      });
      Kokkos::View<int*>::HostMirror sizes2 = create_mirror_view(elmDegree);
      for (int l = 0; l < _numElms; l++){
         sizesArray[l] = sizes2(l); //current_size;//causes segfault for some reason (kokkos view error)
      }
    //start copy from b->a
      CabanaM newCabanaM(sizesArray, _numElms);
  }

  private:
    std::size_t _numPtcls;
    std::size_t _numElms;
    std::size_t _numSoa;
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    int *_parentElm; // parent elm for each soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP

