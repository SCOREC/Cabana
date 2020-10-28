#include <Cabana_Types.hpp>
#include <Cabana_Types.hpp>
#include <CabanaM.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <unordered_map>

namespace Test
{
//---------------------------------------------------------------------------//
//// Test Rebuild
void testRebuild() {
  const int deg[2] = {4, 2}; // before: soa0 [ 0 1 2 3 ], soa1 [ 32 33 ]
  const int deg_len = 2;

  using DataTypes = Cabana::MemberTypes<int, int>; // <newParent, id, active>
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);

  const auto capacity = cm.capacity(); 
  auto new_parents = Cabana::slice<0>(cm.aosoa(), "parents");
  auto id = Cabana::slice<1>(cm.aosoa(), "id");
  auto active = Cabana::slice<2>(cm.aosoa(), "active");

  Kokkos::RangePolicy<TEST_EXECSPACE> id_policy( 0, cm.aosoa().size() );
  Kokkos::parallel_for( id_policy,
    KOKKOS_LAMBDA( const int i ) {
      id(i) = i;
  }, "set_id");

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, capacity);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
     if (tuple == 1 && soa == 0){
       new_parents.access(soa,tuple) = 1;
     } else {
       new_parents.access(soa,tuple) = soa;
     }
  }, "set_parent");
  printf("Capacity: %d\n", capacity);

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, capacity);
  Cabana::simd_parallel_for(simd_policy, 
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, id: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), id.access(soa, tuple), active.access(soa,tuple));
    }, "Final_Print");
  
  cm.rebuild(); // after: soa0 [ 0 2 3 ], soa1 [ 1 32 33 ]

  // Check right particles are active
  active = Cabana::slice<2>(cm.aosoa(), "active");
  // don't need to check soa since both same
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> active_policy(0, capacity);
  Cabana::simd_parallel_for(active_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
     if (tuple <= 2) {
       assert( active.access(soa,tuple) == 1 );
     } else if (tuple > 2) {
       assert( active.access(soa,tuple) == 0 );
     }
  }, "check_active");

  // Check right particles in right place
  // map with <id, soa> pairs
  std::unordered_map<int, int> id_parent_check;
  for (int i = 0; i < cm.aosoa().size(); i++ ) {
    if ( (i == 1) && (i / 32 == 0)) {
      id_parent_check.insert( {i, 1} );
    }
    else {
      id_parent_check.insert( {i, i / 32} );
    }
  }

  // Setup views
  new_parents = Cabana::slice<0>(cm.aosoa(), "parents");
  id = Cabana::slice<1>(cm.aosoa(), "id");
  Kokkos::View<int*> parent_check("parent_check", 2*32);
  Kokkos::View<int*>::HostMirror host_parent_check = Kokkos::create_mirror_view(parent_check);

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> id_check_policy(0, capacity);
  Cabana::simd_parallel_for(id_check_policy, 
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      parent_check((soa*32)+tuple) = id.access(soa, tuple);
  }, "check_id");

  Kokkos::deep_copy(host_parent_check, parent_check);

  for ( int soa = 0; soa <= 1; soa++ ) {
    for ( int tuple = 0; tuple <= 2; tuple++ ) { // <= 2 to ignore inactives
      printf("curr_soa: %d, tuple: %d, id: %d, new_soa: %d\n", soa, tuple, host_parent_check((soa*32)+tuple), id_parent_check[ host_parent_check((soa*32)+tuple) ]);
      assert( id_parent_check[ host_parent_check((soa*32)+tuple) ] == soa );
    }
  }

}

void testBiggerRebuild(){
  printf("\n------- big test -------\n");
  const int deg[3] = {4,2,15}; // [0 1 2 3], [32 33], [64, 65, 66, ..., 78]
  const int deg_len = 3;
  using DataTypes = Cabana::MemberTypes<int, int>;
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);

  const auto capacity = cm.capacity();
  auto new_parents = Cabana::slice<0>(cm.aosoa(), "new_parents");
  auto id = Cabana::slice<1>(cm.aosoa(), "id");
  auto active = Cabana::slice<1>(cm.aosoa(), "active");
  
  Kokkos::RangePolicy<TEST_EXECSPACE> id_policy( 0, cm.aosoa().size() );
  Kokkos::parallel_for( id_policy,
    KOKKOS_LAMBDA( const int i ) {
      id(i) = i;
  }, "set_id");

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, capacity);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) { 
      if (tuple == 1 && soa == 0){ 
        new_parents.access(soa,tuple) = 1;
      }
      else if (tuple == 10 && soa == 2){
        new_parents.access(soa,tuple) = 0;
      }
      else {
        new_parents.access(soa, tuple) = soa;
      }   
      }, "set_parent");
  printf("Capacity: %d\n", capacity);

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, capacity);
  Cabana::simd_parallel_for(simd_policy,
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), active.access(soa,tuple));
    }, "Final_Print");
    
  cm.rebuild(); // [73 0 2 3], [1 32 33], [64, 65, 66, ..., 78]

  // Check right particles are active
  active = Cabana::slice<2>(cm.aosoa(), "active");
  // don't need to check soa since both same
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> active_policy(0, capacity);
  Cabana::simd_parallel_for(active_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
     if (soa == 0) {
       if (tuple <= 3) {
         assert( active.access(soa,tuple) == 1 );
       }
       else {
         assert( active.access(soa,tuple) == 0 );
       }
     }
     else if (soa == 1) {
       if (tuple <= 2) {
         assert( active.access(soa,tuple) == 1 );
       }
       else {
         assert( active.access(soa,tuple) == 0 );
       }
     }
     else if (soa  == 2) {
       if (tuple <= 13) {
         assert( active.access(soa,tuple) == 1 );
       }
       else {
         assert( active.access(soa,tuple) == 0 );
       }
     }
  }, "check_active");

  // Check right particles in right place
  // map with <id, soa> pairs
  std::unordered_map<int, int> id_parent_check;
  for (int i = 0; i < cm.aosoa().size(); i++ ) {
    if ( (i == 1) && (i / 32 == 0)) {
      id_parent_check.insert( {i, 1} );
    }
    else if ( (i == 2*32+10) && (i / 32 == 2)) {
      id_parent_check.insert( {i, 0} );
    }
    else {
      id_parent_check.insert( {i, i / 32} );
    }
  }
  
  // Setup views
  new_parents = Cabana::slice<0>(cm.aosoa(), "parents");
  id = Cabana::slice<1>(cm.aosoa(), "id");
  Kokkos::View<int*> parent_check("parent_check", 3*32);
  Kokkos::View<int*>::HostMirror host_parent_check = Kokkos::create_mirror_view(parent_check);

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> id_check_policy(0, capacity);
  Cabana::simd_parallel_for(id_check_policy, 
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      parent_check((soa*32)+tuple) = id.access(soa, tuple);
  }, "check_id");

  Kokkos::deep_copy(host_parent_check, parent_check);

  for ( int soa = 0; soa <= 2; soa++ ) {
    for ( int tuple = 0; (soa == 0 && tuple <= 3) || (soa == 1 && tuple <= 1) || (soa == 2 && tuple <= 13); tuple++ ) {
      printf("curr_soa: %d, tuple: %d, id: %d, new_soa: %d\n", soa, tuple, host_parent_check((soa*32)+tuple), id_parent_check[ host_parent_check((soa*32)+tuple) ]);
      assert( id_parent_check[ host_parent_check((soa*32)+tuple) ] == soa );
    }
  }
}

TEST( TEST_CATEGORY, aosoa_test )
{
  testRebuild();
  testBiggerRebuild();
}
}
