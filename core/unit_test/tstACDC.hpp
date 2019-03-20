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
#include <Cabana_ACDC.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <stdio.h> // FIXME: remove after debugging

namespace Test
{
//---------------------------------------------------------------------------//
// Test an ACDC
void testACDC()
{

  // Data dimensions.
  const int dim_1 = 3;
  const int dim_2 = 2;
  const int dim_3 = 4;

  // deg array
  const int deg[3] = {4, 1025, 0};
  const int total_particles = deg[0] + deg[1] + deg[2];
  const int deg_len = 3;

  // Declare data types.
  using DataTypes =
      Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                              int,
                              double[dim_1],
                              double[dim_1][dim_2]
                              >;
  using ACDC_t = Cabana::ACDC<DataTypes,TEST_MEMSPACE>;
  ACDC_t acdc( deg, deg_len );
  // TODO: maybe replaced with slicing?
  int vector_len = acdc._vector_length;
  // Sanity check of sizes.
  // FIXME total/vec_len isn't quite right needs to be each index of deg divided
  // by vector len and added but with one small then one large entry it works
  EXPECT_EQ( acdc._size, total_particles/vector_len + deg_len );
  EXPECT_EQ( acdc._aosoa->size(), total_particles/vector_len + deg_len );
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
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_test )
{
    testACDC();
}

//---------------------------------------------------------------------------//

} // end namespace Test
