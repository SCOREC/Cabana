#include <Cabana_Types.hpp>
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
//// Test Rebuild
void testRebuild() {
  const int deg[2] = {4, 2};
  const int deg_len = 2;

  using DataTypes = Cabana::MemberTypes<int>;
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);

  auto new_parents = Cabana::slice<0>(*(cm.aosoa()), "parents");
  auto active = Cabana::slice<1>(cm.aosoaRef(), "active");

  auto capacity = cm.capacity(); 
  printf("capacity of aosoa %d\n", capacity);

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, capacity);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
     if (tuple == 1 && soa == 0){
       new_parents.access(soa,tuple) = 1;
     } else if (soa == 1){
       new_parents.access(soa,tuple) = 1;
     } else {
       new_parents.access(soa, tuple) = soa;
     }
  }, "set_parent");

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, capacity);
  Cabana::simd_parallel_for(simd_policy, 
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), active.access(soa,tuple));
    }, "Final_Print");
  
  cm.rebuild();
}

void testBiggerRebuild(){
  printf("\n------- big test -------\n");
  const int deg[3] = {4, 2,15};
  const int deg_len = 3;
  using DataTypes = Cabana::MemberTypes<int>;
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);

  const auto capacity = cm.capacity();
  auto new_parents = Cabana::slice<0>(cm.aosoaRef(), "new_parents");
  auto new_actives = Cabana::slice<1>(cm.aosoaRef(), "new_actives");

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, capacity);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) { 
      if (tuple == 1 && soa == 0){ 
        new_parents.access(soa,tuple) = 1;
      }
      else if (tuple == 10 && soa == 2){
        new_parents.access(soa,tuple) = 0;
      }
      else if (soa == 1){ 
        new_parents.access(soa,tuple) = 1;
      }
      else if (soa == 2){
        new_parents.access(soa,tuple) = 2;
      }
      else {
        new_parents.access(soa, tuple) = soa;
      }   
      }, "set_parent");
  printf("Capacity: %d\n", capacity);
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, capacity);
  Cabana::simd_parallel_for(simd_policy,
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), new_actives.access(soa,tuple));
    }, "Final_Print");
  cm.rebuild();
}

TEST( TEST_CATEGORY, aosoa_test )
{
  testRebuild();
  testBiggerRebuild();
}
}
