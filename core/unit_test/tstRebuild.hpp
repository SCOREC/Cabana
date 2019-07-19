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

  using DataTypes = Cabana::MemberTypes<int,int>;
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);
  cm.aosoa()->resize(64);

  const auto numPtcls = cm.size();
  auto new_parents = cm.aosoa()->slice<0>();
  auto old_parents = cm.aosoa()->slice<1>();
  auto new_actives = cm.aosoa()->slice<2>();

  int new_parents_d[64];
  for (int pos = 0; pos < 64; pos++){
    if (pos < 32){
      new_parents_d[pos] = 0;
    }
    else {
      new_parents_d[pos] = 1;
    }
  }
  
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, 64);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
      if (tuple == 1 && soa == 0){
        new_parents.access(soa,tuple) = 1;
      }
      else if (soa == 1){
        old_parents.access(soa,tuple) = 1;
        new_parents.access(soa,tuple) = 1;
      }
      else {
        new_parents.access(soa, tuple) = soa;
      }
      }, "set_parent");

    Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> active_policy(0, 64);
    Cabana::simd_parallel_for(active_policy,
      KOKKOS_LAMBDA( const int soa, const int tuple ) {
        if ((tuple < 4 && soa == 0) || (soa == 1 && tuple<2) ){
          new_actives.access(soa, tuple) = 1;
        }
        else {
          new_actives.access(soa, tuple) = 0;
        }
        }, "set_active");
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, 64);
  Cabana::simd_parallel_for(simd_policy, 
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, Old Parent: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), old_parents.access(soa, tuple),  new_actives.access(soa,tuple));
    }, "Final_Print");
  
  cm.rebuild(new_parents_d);
  }

void testBiggerRebuild(){
  const int deg[3] = {4, 2,16};
  const int deg_len = 3;
  const int totalSize = 96;
  using DataTypes = Cabana::MemberTypes<int,int>;
  using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
  using CabanaM_t = Cabana::CabanaM<DataTypes,TEST_MEMSPACE>;
  CabanaM_t cm(deg, deg_len);
  cm.aosoa()->resize(96);

  const auto numPtcls = cm.size();
  auto new_parents = cm.aosoa()->slice<0>();
  auto old_parents = cm.aosoa()->slice<1>();
  auto new_actives = cm.aosoa()->slice<2>();

  int new_parents_d[totalSize];
  for (int pos = 0; pos < totalSize; pos++){
    if (pos < 32){
      new_parents_d[pos] = 0;
    }   
    else if (pos < 64){
      new_parents_d[pos] = 1;
    }   
    else{
      new_parents_d[pos] = 2;
    }
  }
  
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> parent_policy(0, totalSize);
  Cabana::simd_parallel_for(parent_policy, 
    KOKKOS_LAMBDA( const int soa, const int tuple ) { 
      if (tuple == 1 && soa == 0){ 
        new_parents.access(soa,tuple) = 1;
      }
      else if (tuple == 10 && soa == 2){
        new_parents.access(soa,tuple) = 0;
      }
      else if (soa == 1){ 
        old_parents.access(soa,tuple) = 1;
        new_parents.access(soa,tuple) = 1;
      }
      else if (soa == 2){
        old_parents.access(soa,tuple) = 2;
        new_parents.access(soa,tuple) = 2;
      }
      else {
        new_parents.access(soa, tuple) = soa;
      }   
      }, "set_parent");

  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> active_policy(0, totalSize);
  Cabana::simd_parallel_for(active_policy,
    KOKKOS_LAMBDA( const int soa, const int tuple ) {
      if ((tuple < 4 && soa == 0) || (soa == 1 && tuple<2) || (soa == 2 && tuple < 16)){
        new_actives.access(soa, tuple) = 1;
      }
      else {
        new_actives.access(soa, tuple) = 0;
      }
      }, "set_active");
  Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE> simd_policy(0, totalSize);
  Cabana::simd_parallel_for(simd_policy,
    KOKKOS_LAMBDA(const int soa, const int tuple) {
      printf("SoA: %d, Tuple: %d, New Parent: %d, Old Parent: %d, Active: %d\n", soa, tuple, new_parents.access(soa, tuple), old_parents.access(soa, tuple),  new_actives.access(soa,tuple));
    }, "Final_Print");
  cm.rebuild(new_parents_d);
  }

TEST( TEST_CATEGORY, aosoa_test )
{
    testRebuild();
    testBiggerRebuild();
}
}
