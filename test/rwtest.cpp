#include "../fc_btree.h"
#include <fstream>
#include <iostream>

int main() {
  namespace fc = frozenca;
  fc::BTreeSet<int> btree_out;

  constexpr int n = 100;

  for (int i = 0; i < n; ++i) {
    btree_out.insert(i);
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
    if (!btree_in.contains(i)) {
      std::cout << "deserialized tree key lookup failed\n";
    }
  }
  std::cout << "OK\n";
}
