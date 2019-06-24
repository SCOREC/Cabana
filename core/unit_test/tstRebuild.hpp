#include <Cabana_Types.hpp>
#include <CabanaM.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
//// Test an CabanaM
void testRebuild() {
  const int deg[2] = {4, 1024};
  const int deg_len = 2;

  using DataTypes = Cabana::MemberTypes<int,int>; //slice 0 gives new parent elemet id (input)

  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm( deg, deg_len );

  auto slice_float = cm.aosoa()->slice<0>();
  auto slice_int = cm.aosoa()->slice<1>();
  const auto numPtcls = cm.size();

  Kokkos::View<int*,Kokkos::HostSpace> offsets_h("offsets_host",deg_len);
  for (int i=0; i<deg_len; i++)
    offsets_h(i) = cm.offset(i);
  auto offsets_d = Kokkos::create_mirror_view_and_copy(
      TEST_MEMSPACE(), offsets_h);

  printf("numPtcls %d\n", numPtcls);
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy( 0, numPtcls );
  Cabana::simd_parallel_for(simd_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
      auto parentElm = -1;
      printf("soa a parentElm %4d %4d %4d\n", soa, tuple, parentElm);
    }, "foo");
  
  Cabana::Tuple<DataTypes> change = cm.aosoa()->getTuple(0);
/*    for ( int i = 0; i < 4; ++i ) {
      Cabana::get<0>(change,i) = 5;
    }
  */  Cabana::get<1>(change, 0) = 7;

  cm.aosoa()->setTuple(1, change);  
  cm.rebuild();
  
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy2( 0, numPtcls );
  Cabana::simd_parallel_for(simd_policy2,
    KOKKOS_LAMBDA( const int soa2, const int tuple2 ) {
      printf("soa a parentElm %4d %4d %4d\n", soa2, tuple2, parentElm2);
    }, "foo");
}
TEST( TEST_CATEGORY, aosoa_test )
{
    testRebuild();
}
}
