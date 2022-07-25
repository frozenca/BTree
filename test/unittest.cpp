#include "../fc_btree.h"
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
    std::cout << "std::initializer_list test\n";

    fc::BTreeSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
    if (btree.size() == 7) {
      std::cout << "OK\n";
    } else {
      std::cout << "std::initializer_list test failed\n";
    }
  }
  {
    std::cout << "Multiset test\n";

    fc::BTreeMultiSet<int> btree{1, 4, 3, 2, 3, 3, 6, 5, 8};
    if (btree.size() == 9) {
      std::cout << "OK\n";
    } else {
      std::cout << "Multiset test failed\n";
    }
    btree.erase(3);
    if (btree.size() == 6) {
      std::cout << "OK\n";
    } else {
      std::cout << "Multiset test failed\n";
    }
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
    auto rg = btree.enumerate(20, 30);
    if (std::ranges::distance(rg.begin(), rg.end()) == 11) {
      std::cout << "OK\n";
    } else {
      std::cout << "Enumerate failed\n";
    }

    std::cout << "erase_if test\n";
    btree.erase_if([](auto n) { return n >= 20 && n <= 90; });
    if (btree.size() == 29) {
      std::cout << "OK\n";
    } else {
      std::cout << "erase_if failed\n";
    }
  }
  {
    std::cout << "BTreeMap test\n";
    fc::BTreeMap<std::string, int> btree;

    btree["asd"] = 3;
    btree["a"] = 6;
    btree["bbb"] = 9;
    btree["asdf"] = 8;
    btree["asdf"] = 333;
    if (btree["asdf"] == 333) {
      std::cout << "OK\n";
    } else {
      std::cout << "operator[] failed\n";
    }

    btree.emplace("asdfgh", 200);
    if (btree["asdfgh"] == 200) {
      std::cout << "OK\n";
    } else {
      std::cout << "emplace() failed\n";
    }
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
    if (btree5.contains(200)) {
      std::cout << "Split failed\n";
    }
    for (int i = 201; i < 300; ++i) {
      if (!btree5.contains(i)) {
        std::cout << "Split failed\n";
      }
    }
    std::cout << "Multiset split test\n";

    fc::BTreeMultiSet<int> btree6;
    btree6.insert(0);
    btree6.insert(2);
    for (int i = 0; i < 100; ++i) {
      btree6.insert(1);
    }
    auto [btree7, btree8] = fc::split(std::move(btree6), 1);
    if (btree7.size() != 1 || btree8.size() != 1) {
      std::cout << "Split failed: " << btree7.size() << ' ' << btree8.size()
                << '\n';
    }

    std::cout << "OK\n";
  }
  {
    std::cout << "Two arguments join test\n";
    fc::BTreeSet<int> tree1;
    for (int i = 0; i < 100; ++i) {
      tree1.insert(i);
    }
    fc::BTreeSet<int> tree2;
    for (int i = 100; i < 200; ++i) {
      tree2.insert(i);
    }
    auto tree3 = fc::join(std::move(tree1), std::move(tree2));
    for (int i = 0; i < 200; ++i) {
      if (!tree3.contains(i)) {
        std::cout << "Join fail\n";
      }
    }
    std::cout << "OK\n";
  }
  {
    std::cout << "Three arguments split test\n";
    fc::BTreeSet<int> tree1;
    for (int i = 0; i < 100; ++i) {
      tree1.insert(i);
    }
    auto [tree2, tree3] = fc::split(std::move(tree1), 10, 80);
    if (tree2.size() != 10 || tree3.size() != 19) {
      std::cout << "Split fail\n";
    }
    std::cout << "OK\n";
  }
  {
    std::cout << "Multiset erase test\n";
    fc::BTreeMultiSet<int> tree1;
    tree1.insert(0);
    for (int i = 0; i < 100; ++i) {
      tree1.insert(1);
    }
    tree1.insert(2);

    tree1.erase(1);

    if (tree1.size() != 2) {
      std::cout << "Mutliset erase failed: " << tree1.size() << '\n';
    }
    std::cout << "OK\n";
  }
  {
    std::cout << "Range insert test\n";
    fc::BTreeSet<int> btree;
    btree.insert(1);
    btree.insert(10);

    std::vector<int> v{2, 5, 4, 3, 7, 6, 6, 6, 2, 8, 8, 9};
    btree.insert_range(std::move(v));

    for (int i = 1; i < 10; ++i) {
      if (!btree.contains(i)) {
        std::cout << "Range insert failed\n";
      }
    }
    std::cout << "OK\n";
  }
  {
    fc::BTreeSet<int> btree;
    btree.insert(1);
    btree.insert(10);

    std::vector<int> v{2, 5, 4, 3, 7, 6, 6, 6, 2, 8, 8, 9, 10};
    btree.insert_range(std::move(v));

    for (int i = 1; i < 10; ++i) {
      if (!btree.contains(i)) {
        std::cout << "Range insert failed\n";
      }
    }
    std::cout << "OK\n";
  }
  {
    std::cout << "count() test\n";
    fc::BTreeMultiSet<int> btree2;
    btree2.insert(1);
    btree2.insert(1);
    if (btree2.count(1) != 2 || btree2.count(0) != 0 || btree2.count(2) != 0) {
      std::cout << "count() failed\n";
    }
    std::cout << "OK\n";
  }
}
