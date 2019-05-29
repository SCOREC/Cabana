
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

  private:
    std::size_t _size; // size of offset array
    std::size_t _vector_length;
    int *_offsets; // offset array for soa
    AoSoA_t *_aosoa; // pointer to AoSoA

};

} // end namespace Cabana

#endif // CABANAM_HPP
