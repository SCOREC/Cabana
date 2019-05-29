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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <stdio.h> // FIXME: remove after debugging

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
  /*
  printf("AoSoA vector length: %d\n", vector_len);
  printf("AoSoA size: %d\n", cabanam.size());
  printf("AoSoA numSoA: %d\n", cabanam._aosoa->numSoA());
  */
  int offset1 = (deg[0]/vector_len) + 1;
  int offset2 = (deg[1]/vector_len) + offset1 + 1;
  int offset3 = offset2 + 1; // should have a single empty SoA
  int test_offsets[4] = {0, offset1, offset2, offset3};
  for (int i=0;i<4;++i) {
    EXPECT_EQ( cabanam.offset(i), test_offsets[i] );
}
/*
void testData()
{

  const int deg[3] = {4, 1024, 0};
  const int deg_len = 3;

  using DataTypes =
      Cabana::MemberTypes<float,int>;

  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cabanam( deg, deg_len );

  auto slice_0 = cabanam.aosoa().slice<0>();
  float *p_float = slice_0.data();
  int *p_int = cabanam.aosoa().slice<1>().data();
  printf("%f %d\n", p_float[0], p_int[0]);

  // TODO get stride lengths
  // incorporate stride lengths into offet
  // figure out how to iterate using offset
  // maybe use access to get an SoA
  for (int p=0; p<cabanam.offset(i); ++p) {
  }
  //acdc._aosoa->access( 0 );
}
*/
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_test )
{
    testCabanaM();
}

//---------------------------------------------------------------------------//

} // end namespace Test
