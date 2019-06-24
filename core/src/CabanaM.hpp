
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
      for ( int i=0; i<elem_count; ++i ) {
        int SoA_count = (deg[i]/_vector_length) + 1;
        _offsets[i+1] = SoA_count + _offsets[i];
      }

      _size = _offsets[elem_count];
      _aosoa->resize( _size );
    }

    KOKKOS_FUNCTION
    std::size_t size() const { return _size; }

    KOKKOS_FUNCTION
    std::size_t vector_length() const { return _vector_length; }

    KOKKOS_FUNCTION
    int offset(int i) const { return _offsets[i]; }

    KOKKOS_FUNCTION
    AoSoA_t* aosoa() { return _aosoa; }

   void rebuild(/*int* new_parent, int num_new_ptcls, int* new_ptcl_parents*/)
//,SOA* new_ptcl)
    {
      const int SIMD_WIDTH = 32/*vector_length()*/; //cant be from a function call
      
      int numSoA = this->aosoa()->numSoA();
      Kokkos::View<int*> sizes("new_sizes", numSoA);
      Kokkos::View<int*> newOffsets("new_offsets",size());
      auto slice_int = slice<0>(*aosoa()); //user input of destination
      using ExecutionSpace = Kokkos::DefaultExecutionSpace;
      //first loop to count number of particles per element (atomic)
      auto atomic = KOKKOS_LAMBDA(const int& i,const int& a){
        Kokkos::atomic_increment<int>(&sizes[slice_int.access(i,a)]);
      };
      Cabana::SimdPolicy<SIMD_WIDTH,ExecutionSpace> simd_policy( 0, size()  ); 
      Cabana::simd_parallel_for( simd_policy, atomic, "atomic" );
      //kokkos parllel print 
      //movee 1 particle
      //print offset

      /*Kokkos::parallel_for("particlesPerElement", size(), KOKKOS_LAMBDA(const int& i,const int& a){
	ptr = &sizes[slice_int.access(i,a)];
        Kokkos::atomic_increment<int>(ptr);
      });*/
/*      CabanaM newCabanaM(sizes, size() );//host view
      //fill by usig offset array to find (in the old cabana)  all the particles for an element, then store them next to each other in the new cabana
      Kokkos::parallel_scan(numSoA, newOffsets, KOKKOS_LAMBDA( const int& i, int& upd, const bool& last) {
        const int val_i = offset(i);
        if (last){
	 newCabanaM._offsets[i] = upd;
        } 
	upd+= val_i;
      });
      auto BtoA = KOKKOS_LAMBDA(int i, int a){
        int b = slice_int.access(i, a);//returns bool fix
        auto first = newCabanaM._offsets(b);//new offsets
        auto j = Kokkos::atomic_fetch_add<int>(&sizes[slice_int.access(i,a)], 1);
        auto tp= aosoa().getTuple(a);
        newCabanaM->aosoa()->setTuple(a,tp);
        };
      Cabana::SimdPolicy<SIMD_WIDTH,ExecutionSpace> BtoA_policy( 0, numSoA);
      Cabana::simd_parallel_for( simd_policy, BtoA_policy, "BtoA" );
    free(this);
 */ }

  private:
    std::size_t _size; // size of offset array
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP

