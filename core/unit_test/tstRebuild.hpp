#include <Cabana_Types.hpp>
#include <CabanaM.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <stdio.h> // FIXME: remove after debugging

namespace Test
{
//---------------------------------------------------------------------------//
//// Test an CabanaM
void testRebuild() {
printf("I hope this works!");


  const int dim_1 = 3;
  const int dim_2 = 2;
  const int dim_3 = 4;

  const int deg[3] = {4, 1024, 0};
  const int deg_len = 3;

 using DataTypes =
      Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                              int,
                              double[dim_1],
                              double[dim_1][dim_2]
                              >;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t acdc( deg, deg_len );
 int vector_len = acdc._vector_length;
  int ex_size = deg[0]/vector_len+deg[1]/vector_len+deg[2]/vector_len+deg_len;
EXPECT_EQ( acdc._size, ex_size );
  printf("AoSoA vector length: %d\n", vector_len);
  printf("AoSoA size: %d\n", acdc._aosoa->size());
  printf("AoSoA numSoA: %d\n", acdc._aosoa->numSoA());
  int offset1 = (deg[0]/vector_len) + 1;
  int offset2 = (deg[1]/vector_len) + offset1 + 1;
  int offset3 = offset2 + 1; // should have a single empty SoA
  int test_offsets[4] = {0, offset1, offset2, offset3};
  for (int i=0;i<4;++i) {
    EXPECT_EQ( acdc._offsets[i], test_offsets[i] );
  }
  acdc._aosoa->access( 0 );
  rebuild();
  printf("after rebuild:\n")
  EXPECT_EQ( acdc._size, ex_size );
  printf("AoSoA vector length: %d\n", vector_len);
  printf("AoSoA size: %d\n", acdc._aosoa->size());
  printf("AoSoA numSoA: %d\n", acdc._aosoa->numSoA());
for (int i=0;i<4;++i) {
    EXPECT_EQ( acdc._offsets[i], test_offsets[i] );
  }
  acdc._aosoa->access( 0 );
}
TEST_F( TEST_CATEGORY, aosoa_test )
{
    testRebuild();
}
}
