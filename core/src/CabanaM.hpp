
#ifndef CABANAM_HPP
#define CABANAM_HPP

#include <Cabana_MemberTypes.hpp>
#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Distributor.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
/* class CabanaM */
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

    void rebuild()//int* new_parent, int num_new_ptcls, int* new_ptcl_parents)//, 
//SOA* new_ptcl)
    { //simd = vector length
      int SIMD_WIDTH = ;
      int count = (num_new_ptcls/SIMD_WIDTH) + 1;
      int numAoSoA = num_new_ptcls - (SIMD_WIDTH * (count - 1));
      int sizes[count] = {SIMD_WIDTH};
      sizes[count] = numAoSoA;
	auto dst = new AoSoA_t();
     // AoSoA<typename CM_DT::type,MemorySpace> newAoSoA("src",count);
     // AoSoA<typename CM_DT::type,MemorySpace> newAoSoA("src",count);
     // auto copy = KOKKOS_LAMBDA(int s){
/*        auto& soa = aosoa().access(s);
	auto& newSoa = newAosoa().access(s);
        for ( int i = 0; i < 3; ++i )
          for ( int j = 0; j < 3; ++j )
            for ( unsigned a = 0; a < aosoa().arraySize(s); ++a )
              Cabana::get<0>(newSoa,a,i,j) = Cabana::get<0>(soa,a,i,j);

        for ( int i = 0; i < 4; ++i )
          for ( unsigned a = 0; a < aosoa().arraySize(s); ++a ) 

        for ( unsigned a = 0; a < aosoa().arraySize(s); ++a )
          Cabana::get<2>(newSoa,a) = Cabana::get<2>(soa,a);*/
        //deep_copy(newAoSoA, aosoa);
       deep_copy( dst, aosoa() );
    //  };
      //SimdPolicy<vector_length(),MemorySpace> simd_policy( 0, count-1 );
     //simd_parallel_for(simd_policy, copy, "rebuild");
     free(dst);
     //lambda use: https://github.com/SCOREC/Cabana/blob/830dfbcaa0a3dfb3c66a888c86e765f05ec14ebc/core/unit_test/tstHalo.hpp
     //
    }


  private:
    std::size_t _size; // size of offset array
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP
