#define _UNIT_TEST

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>

#include "fc_btree.h"

TEST_CASE("rw-test") {
  namespace fc = frozenca;
  fc::BTreeSet<int> btree_out;

  constexpr int n = 100;

  for (int i = 0; i < n; ++i) {
    REQUIRE_NOTHROW(btree_out.insert(i));
  }
  {
    std::ofstream ofs{"btree.bin", std::ios_base::out | std::ios_base::binary |
                                       std::ios_base::trunc};
    ofs << btree_out;
  }

  fc::BTreeSet<int> btree_in;
  {
    std::ifstream ifs{"btree.bin", std::ios_base::in | std::ios_base::binary};
    ifs >> btree_in;
  }

  for (int i = 0; i < n; ++i) {
    REQUIRE(btree_in.contains(i));
  }
}
