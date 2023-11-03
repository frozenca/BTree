#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "fc_catch2.h"
#include "fc/disk_btree.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>
#include <unordered_map>

struct stats {
  float average = 0.0f;
  float stdev = 0.0f;
  float percentile_95 = 0.0f;
  float percentile_99 = 0.0f;
  float percentile_999 = 0.0f;
};

stats get_statistics(std::vector<float> &v) {
  auto n = std::ssize(v);
  if (n == 0) {
    return {};
  }
  stats s;
  s.average = std::accumulate(v.begin(), v.end(), 0.0f) / n;
  float variance = 0.0f;
  for (auto value : v) {
    variance += std::pow(value - s.average, 2.0f);
  }
  variance /= n;
  s.stdev = std::sqrt(variance);
  std::ranges::sort(v);
  s.percentile_95 = *(v.begin() + (19 * n / 20));
  s.percentile_99 = *(v.begin() + (99 * n / 100));
  s.percentile_999 = *(v.begin() + (999 * n / 1000));
  return s;
}

struct perf_result {
  size_t values_cnt{0};
  stats insert;
  stats find;
  stats erase;
  void print_stats() const {
    auto print = [this](const std::string &stat_name, const stats &stat) {
      std::cout << "\tTime to " << stat_name << " " << values_cnt << " elements:\n"
                << "\tAverage : " << stat.average << "ms,\n"
                << "\tStdev   : " << stat.stdev << "ms,\n"
                << "\t95%     : " << stat.percentile_95 << "ms,\n"
                << "\t99%     : " << stat.percentile_99 << "ms,\n"
                << "\t99.9%   : " << stat.percentile_999 << "ms,\n";
    };
    print("insert", insert);
    print("find", find);
    print("erase", erase);
  }
};

template <typename TreeType>
[[maybe_unused]] perf_result tree_perf_test(TreeType &tree, size_t values_cnt = 1'000'000, size_t trials = 1) {
  const size_t max_n = values_cnt;
  const size_t max_trials = trials;

  std::mt19937 gen(std::random_device{}());
  std::vector<float> durations_insert;
  std::vector<float> durations_find;
  std::vector<float> durations_erase;
  std::vector<typename TreeType::value_type> v(max_n);
  std::iota(v.begin(), v.end(), 0);

  for (size_t t = 0; t < max_trials; ++t) {
    float duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      tree.insert(num);
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_insert.push_back(duration);

    duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      if (!tree.contains(num)) {
        std::cerr << "Lookup verification fail!\n";
      }
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_find.push_back(duration);

    duration = 0.0f;
    std::ranges::shuffle(v, gen);
    for (auto num : v) {
      auto start = std::chrono::steady_clock::now();
      if (!tree.erase(num)) {
        std::cerr << "Erase verification fail!\n";
      }
      auto end = std::chrono::steady_clock::now();
      duration += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();
    }
    durations_erase.push_back(duration);
  }
  perf_result result;
  result.values_cnt = max_n;
  result.insert = get_statistics(durations_insert);
  result.find = get_statistics(durations_find);
  result.erase = get_statistics(durations_erase);
  return result;
}

TEST_CASE("perftest") {
  namespace fc = frozenca;
  std::unordered_map<std::string, perf_result> result;

  BENCHMARK("Balanced tree test - warmap") {
    fc::BTreeSet<std::int64_t> btree;
    tree_perf_test(btree);
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 64)") {
    fc::BTreeSet<std::int64_t> btree;
    result.emplace("BTreeSet(64)", tree_perf_test(btree));
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 96)") {
    fc::BTreeSet<std::int64_t, 96> btree;
    result.emplace("BTreeSet(96)", tree_perf_test(btree));
  };
  BENCHMARK("frozenca::DiskBTreeSet test (fanout 128)") {
    fc::DiskBTreeSet<std::int64_t, 128> btree("database.bin", 1UL << 25UL, true);
    result.emplace("DiskBTreeSet(128)", tree_perf_test(btree));
  };
  BENCHMARK("frozenca::BTreeSet test (fanout 128)") {
    fc::BTreeSet<std::int64_t, 128> btree;
    result.emplace("BTreeSet(128)", tree_perf_test(btree));
  };
  BENCHMARK("frozenca::BTreeSet test (don't use SIMD)") {
    fc::BTreeSet<std::uint64_t> btree;
    result.emplace("BTreeSet(no-simd)", tree_perf_test(btree));
  };
  BENCHMARK("std::set test") {
    std::set<std::int64_t> rbtree;
    result.emplace("std::set", tree_perf_test(rbtree));
  };

  for (const auto &[key, value] : result) {
    std::cout << "----------------" << '\n';
    std::cout << key << '\n';
    value.print_stats();
  }
}
