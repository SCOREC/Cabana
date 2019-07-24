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

  public:
    CabanaM()
      : _size( 0 )
      , _vector_length( 0 )
      , _offsets( NULL )
      , _aosoa( NULL )
    {}

    CabanaM( const int *deg, const int elem_count )
    {
      _aosoa = new AoSoA_t();
      //int vector_len = _aosoa->vector_length;
      _vector_length = Impl::PerformanceTraits<
             typename MemorySpace::execution_space>::vector_length;
      _offsets = new int[elem_count+1]();
      // elem at i owns SoA offsets[i+1] - offsets[i]
      _offsets[0] = 0;
      for ( int i=0; i<elem_count; ++i ) {
        int SoA_count = (deg[i]/_vector_length) + 1;
        _offsets[i+1] = SoA_count + _offsets[i];
      }
      _size = _offsets[elem_count] * _vector_length;
      _aosoa->resize( _size); 
    }

    KOKKOS_FUNCTION
    std::size_t size() const { return _size; }

    KOKKOS_FUNCTION
    std::size_t vector_length() const { return _vector_length; }

    KOKKOS_FUNCTION
    int offset(int i) const { return _offsets[i]; }

    KOKKOS_FUNCTION
    AoSoA_t* aosoa() { return _aosoa; }

   void rebuild(int* new_parent/*, int num_new_ptcls, int* new_ptcl_parents*/)
//,SOA* new_ptcl)
    {
      const int SIMD_WIDTH = 32/*vector_length()*/; //cant be from a function call
      int numSoA = this->aosoa()->numSoA();
      int sizesArray[numSoA];
      Kokkos::View<int*> sizes("new_sizes", numSoA);
      Kokkos::View<int*> newOffsets("new_offsets", numSoA);
      auto active = slice<2>(*aosoa());
      using ExecutionSpace = Kokkos::DefaultExecutionSpace;
      //first loop to count number of particles per element (atomic)
      auto atomic = KOKKOS_LAMBDA(const int& i,const int& a){
        if (active.access(i,a) == 1){
          Kokkos::atomic_increment<int>(&sizes(i));
        }
      };
      Cabana::SimdPolicy<SIMD_WIDTH,ExecutionSpace> simd_policy( 0, size()); 
      Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
      Kokkos::parallel_for(numSoA, KOKKOS_LAMBDA(const int i){
         auto current_size = sizes(i);
         printf("Degree of %d at position %d\n", current_size, i); 
      });
      newOffsets = sizes;
      Kokkos::parallel_scan (numSoA, KOKKOS_LAMBDA (const int& i, int& upd, const bool &last) {
          const int val_i = newOffsets(i);
          if (last){
            newOffsets(i) = upd;
          } 
          upd+= val_i;
      });
      
      Kokkos::parallel_for(numSoA, KOKKOS_LAMBDA(const int i){
        auto current_offset = newOffsets(i);
        printf("Offset of %d at position %d\n", current_offset, i);
      });
      Kokkos::View<int*>::HostMirror sizes2 = create_mirror_view(sizes);
      for (int l = 0; l < numSoA; l++){
        // int current_size = sizes(i);
         sizesArray[l] = sizes2(l); //current_size;//causes segfault for some reason (kokkos view error)
      }
    //start copy from b->a
      CabanaM newCabanaM(sizesArray, numSoA);
      auto slice0 = slice<0>(*newCabanaM.aosoa());
      auto slice1 = slice<1>(*newCabanaM.aosoa());
      auto slice2 = slice<2>(*newCabanaM.aosoa());
      auto oldSlice0 = slice<0>(*aosoa());
      auto oldSlice1 = slice<1>(*aosoa());
      auto oldSlice2 = slice<2>(*aosoa());
      //fill by usig offset array to find (in the old cabana)  all the particles for an element, then store them next to each other in the new cabana
      auto BtoA = KOKKOS_LAMBDA(const int& i, const int& a){
    //    int b = slice_int.access(i, a);//returns bool fix
   //     auto first = newCabanaM._offsets(b);//new offsets
   //     auto j = Kokkos::atomic_fetch_add<int>(&sizes[slice_int.access(i,a)], 1);
    //    auto tp= aosoa().getTuple(a);
          slice0.access(i,a) = oldSlice0.access(i,a);
          slice1.access(i,a) = oldSlice1.access(i,a);
          slice2.access(i,a) = oldSlice2.access(i,a);
        //newCabanaM->aosoa()->setTuple(a,tp);
        };
      Cabana::SimdPolicy<SIMD_WIDTH,ExecutionSpace> BtoA_policy( 0, numSoA);
      Cabana::simd_parallel_for( BtoA_policy, BtoA, "BtoA" );
     auto printBtoA = KOKKOS_LAMBDA(const int& i, const int& a){
       printf("For slice 0, SoA: %d Tuple: %d Old Value: %d New Value: %d\n", i, a, oldSlice0.access(i,a), slice0.access(i,a));
       printf("For slice 1, SoA: %d Tuple: %d Old Value: %d New Value: %d\n", i, a, oldSlice1.access(i,a), slice1.access(i,a));
       printf("For slice 2, SoA: %d Tuple: %d Old Value: %d New Value: %d\n", i, a, oldSlice2.access(i,a), slice2.access(i,a));
     };
     Cabana::simd_parallel_for(BtoA_policy, printBtoA, "BtoA_print");
  }

  private:
    std::size_t _size; // size of offset array
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP

