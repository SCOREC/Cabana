
#ifndef CABANA_ACDC_HPP
#define CABANA_ACDC_HPP

#include <Cabana_MemberTypes.hpp>
//#include <Cabana_Slice.hpp>
//#include <Cabana_Tuple.hpp>
#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
//#include <Cabana_SoA.hpp>
//#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>

#include <Kokkos_Core.hpp>


namespace Cabana
{
/* class ACDC */
template<class DataTypes,
         class MemorySpace>
class ACDC
{
  public:

    using AoSoA_t = Cabana::AoSoA<DataTypes,MemorySpace>;

  public:
    ACDC()
      : _size( 0 )
      , _vector_length( 0 )
      , _offsets( NULL )
      , _aosoa( NULL )
    {}

    ACDC( const int *deg, const int elem_count )
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

  //private:
  public: // FIXME: make private again
    int _size; // size of offset array
    int _vector_length;
    int *_offsets; // offset array for soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANA_ACDC_HPP
