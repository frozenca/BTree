#include "fc_btree.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

int main() {
  namespace fc = frozenca;

  // recommend to use debug build for these unit tests

  std::cout << "B-Tree unit tests\n";

  {
    fc::BTreeSet<int> btree;
    constexpr int n = 100;

    std::mt19937 gen(std::random_device{}());

    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);

    std::cout << "Random insert\n";
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      btree.insert(num);
    }
    std::cout << "OK\n";

    std::cout << "Random lookup\n";
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      if (!btree.contains(num)) {
        std::cerr << "Lookup failed!\n";
      }
    }
    std::cout << "OK\n";

    std::cout << "Random erase\n";
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      if (!btree.erase(num)) {
        std::cerr << "Erase failed!\n";
      }
    }
    std::cout << "OK\n";
  }

  {
    std::cout << "std:initializer_list test\n";

    fc::BTreeSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
    for (auto num : btree) {
      std::cout << num << ' ';
    }
    std::cout << '\n';
  }
  {
    std::cout << "Multiset test\n";

    fc::BTreeMultiSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
    for (auto num : btree) {
      std::cout << num << ' ';
    }
    std::cout << '\n';
    btree.erase(3);
    for (auto num : btree) {
      std::cout << num << ' ';
    }
    std::cout << '\n';
  }
  {
    std::cout << "Order statistic test\n";

    fc::BTreeSet<int> btree;
    constexpr int n = 100;

    for (int i = 0; i < n; ++i) {
      btree.insert(i);
    }

    for (int i = 0; i < n; ++i) {
      if (btree.kth(i) != i) {
        std::cout << "Kth failed\n";
      }
    }
    std::cout << "OK\n";

    for (int i = 0; i < n; ++i) {
      if (btree.order(btree.find(i)) != i) {
        std::cout << "Order failed\n";
      }
    }
    std::cout << "OK\n";
  }
  {
    std::cout << "Enumerate test\n";
    fc::BTreeSet<int> btree;
    constexpr int n = 100;

    for (int i = 0; i < n; ++i) {
      btree.insert(i);
    }
    for (auto num : btree.enumerate(20, 30)) {
      std::cout << num << ' ';
    }
    std::cout << '\n';

    std::cout << "erase_if test\n";
    btree.erase_if([](auto n) { return n >= 20 && n <= 90; });
    for (auto num : btree) {
      std::cout << num << ' ';
    }
    std::cout << '\n';
  }
  {
    std::cout << "BTreeMap test\n";
    fc::BTreeMap<std::string, int> btree;

    btree["asd"] = 3;
    btree["a"] = 6;
    btree["bbb"] = 9;
    btree["asdf"] = 8;
    for (const auto &[k, v] : btree) {
      std::cout << k << ' ' << v << '\n';
    }
    std::cout << '\n';

    btree["asdf"] = 333;
    for (const auto &[k, v] : btree) {
      std::cout << k << ' ' << v << '\n';
    }
    std::cout << '\n';

    btree.emplace("asdfgh", 200);
    for (const auto &[k, v] : btree) {
      std::cout << k << ' ' << v << '\n';
    }
    std::cout << '\n';
  }
  {
    std::cout << "join/split test\n";
    fc::BTreeSet<int> btree1;
    for (int i = 0; i < 100; ++i) {
      btree1.insert(i);
    }

    fc::BTreeSet<int> btree2;
    for (int i = 101; i < 300; ++i) {
      btree2.insert(i);
    }

    auto btree3 = fc::join(std::move(btree1), 100, std::move(btree2));

    for (int i = 0; i < 300; ++i) {
      if (!btree3.contains(i)) {
        std::cout << "Join failed\n";
      }
    }
    std::cout << "OK\n";

    auto [btree4, btree5] = fc::split(std::move(btree3), 200);
    for (int i = 0; i < 200; ++i) {
      if (!btree4.contains(i)) {
        std::cout << "Split failed\n";
      }
    }
    for (int i = 201; i < 300; ++i) {
      if (!btree5.contains(i)) {
        std::cout << "Split failed\n";
      }
    }

    std::cout << "OK\n";
  }
}