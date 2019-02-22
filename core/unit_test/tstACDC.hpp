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
  const int deg[2] = {70, 4};

  // Declare data types.
  using DataTypes =
      Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                              int,
                              double[dim_1],
                              double[dim_1][dim_2]
                              >;
  using ACDC_t = Cabana::ACDC<DataTypes,TEST_MEMSPACE>;
  // TODO MODIFY TESTS TO WORK WIHT ARBITRARY VECTOR LENGTH
  ACDC_t acdc( deg, 2 );
  // could be replaced with slicing?
  int vector_len = acdc._vector_length;
  // Check sizes.
  EXPECT_EQ( acdc._size, 74/vector_len+2 );
  EXPECT_EQ( acdc._aosoa->size(), 74/vector_len+2 );
  int offset1 = deg[0]/vector_len+1;
  int offset2 = deg[1]/vector_len+offset1+1;
  int test_offsets[3] = {0, offset1, offset2};
  for (int i=0;i<3;++i)
    EXPECT_EQ( acdc._offsets[i], test_offsets[i] );
  /*
  EXPECT_EQ( acdc._offsets, nullptr );
  EXPECT_EQ( acdc._size, int(0) );
  EXPECT_EQ( acdc._aosoa.size(), int(0) );
  EXPECT_EQ( acdc._aosoa.capacity(), int(0) );
  EXPECT_EQ( acdc._aosoa.numSoA(), int(0) );
  */
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
