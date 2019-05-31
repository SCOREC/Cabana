/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Types.hpp>
#include <CabanaM.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>

#include <Cabana_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Test an CabanaM
void testCabanaM()
{

  // Data dimensions.
  const int dim_1 = 3;
  const int dim_2 = 2;
  const int dim_3 = 4;

  // deg array
  const int deg[3] = {4, 1024, 0};
  const int deg_len = 3;

  // Declare data types.
  using DataTypes =
      Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                              int,
                              double[dim_1],
                              double[dim_1][dim_2]
                              >;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cabanam( deg, deg_len );
  // need vector length because test changes if on device
  int vector_len = cabanam.vector_length();
  int ex_size = deg[0]/vector_len+deg[1]/vector_len+deg[2]/vector_len+deg_len;
  // Sanity check of sizes.
  EXPECT_EQ( cabanam.size(), ex_size );
  int offset1 = (deg[0]/vector_len) + 1;
  int offset2 = (deg[1]/vector_len) + offset1 + 1;
  int offset3 = offset2 + 1; // should have a single empty SoA
  int test_offsets[4] = {0, offset1, offset2, offset3};
  for (int i=0;i<4;++i) {
    EXPECT_EQ( cabanam.offset(i), test_offsets[i] );
  }
}

void testData()
{

  const int deg[3] = {4, 1024, 0};
  const int deg_len = 3;

  using DataTypes = Cabana::MemberTypes<float,int>;

  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm( deg, deg_len );

  auto slice_float = cm.aosoa()->slice<0>();
  auto slice_int = cm.aosoa()->slice<1>();


  Kokkos::View<int*,Kokkos::HostSpace> offsets_h("offsets_host",deg_len);
  for (int i=0; i<deg_len; i++)
    offsets_h(i) = cm.offset(i);
  auto offsets_d = Kokkos::create_mirror_view_and_copy(
      TEST_MEMSPACE(), offsets_h);

  const auto numPtcls = cm.size();
  printf("numPtcls %d\n", numPtcls);
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy( 0, numPtcls );
  Cabana::simd_parallel_for(simd_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
      auto parentElm = -1;
      printf("soa a parentElm %4d %4d %4d\n", soa, tuple, parentElm);
    }, "foo");
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_test )
{
  testCabanaM();
  testData();
}

//---------------------------------------------------------------------------//

} // end namespace Test
