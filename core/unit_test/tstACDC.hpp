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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

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

  // Declare data types.
  using DataTypes =
      Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                              int,
                              double[dim_1],
                              double[dim_1][dim_2]
                              >;
  using ACDC_t = Cabana::ACDC<DataTypes,TEST_MEMSPACE>;
  ACDC_t acdc;
  // Check sizes.
  EXPECT_EQ( acdc._offsets, nullptr );
  EXPECT_EQ( acdc._aosoa, nullptr );
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
