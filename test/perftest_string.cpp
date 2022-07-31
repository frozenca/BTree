#include "../fc_disk_btree.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

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

std::vector<std::string> generate_random_strings(int max_n, int max_length, bool allow_duplicates) {
    std::vector<std::string> res;

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> length_dist (1, max_length);
    std::uniform_int_distribution<int> ch_dist (32, 126);

    for (int i = 0; i < max_n; ++i) {
        int len = length_dist(gen);
        std::string s;
        for (int l = 0; l < len; ++l) {
            s += static_cast<char>(ch_dist(gen));
        }
        res.push_back(std::move(s));
    }

    if (!allow_duplicates) {
        std::ranges::sort(res);
        auto ret = std::ranges::unique(res);
        res.erase(ret.begin(), ret.end());
    }

    return res;
}

constexpr int max_n = 1'000'000;
constexpr int max_length = 50;

auto str_vec = generate_random_strings(max_n, max_length, false);

template <typename TreeType>
void tree_perf_test(TreeType &tree, bool warmup = false) {
    constexpr int max_trials = 10;

    std::mt19937 gen(std::random_device{}());
    std::vector<float> durations_insert;
    std::vector<float> durations_find;
    std::vector<float> durations_erase;
    std::vector<typename TreeType::value_type> v(max_n);


    for (int t = 0; t < max_trials; ++t) {
        float duration = 0.0f;
        std::ranges::shuffle(str_vec, gen);
        for (const auto& str : str_vec) {
            auto start = std::chrono::steady_clock::now();
            tree.insert(str);
            auto end = std::chrono::steady_clock::now();
            duration +=
                    std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
                            end - start)
                            .count();
        }
        durations_insert.push_back(duration);

        duration = 0.0f;
        std::ranges::shuffle(str_vec, gen);
        for (const auto& str : str_vec) {
            auto start = std::chrono::steady_clock::now();
            if (!tree.contains(str)) {
                std::cerr << "Lookup verification fail!\n";
            }
            auto end = std::chrono::steady_clock::now();
            duration +=
                    std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
                            end - start)
                            .count();
        }
        durations_find.push_back(duration);

        duration = 0.0f;
        std::ranges::shuffle(str_vec, gen);
        for (const auto& str : str_vec) {
            auto start = std::chrono::steady_clock::now();
            if (!tree.erase(str)) {
                std::cerr << "Erase verification fail!\n";
            }
            auto end = std::chrono::steady_clock::now();
            duration +=
                    std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
                            end - start)
                            .count();
        }
        durations_erase.push_back(duration);
    }
    if (!warmup) {
        {
            auto stat = get_statistics(durations_insert);
            std::cout << "Time to insert " << max_n << " elements:\n"
                      << "Average : " << stat.average << "ms,\n"
                      << "Stdev   : " << stat.stdev << "ms,\n"
                      << "95%     : " << stat.percentile_95 << "ms,\n"
                      << "99%     : " << stat.percentile_99 << "ms,\n"
                      << "99.9%   : " << stat.percentile_999 << "ms,\n";
        }
        {
            auto stat = get_statistics(durations_find);
            std::cout << "Time to lookup " << max_n << " elements:\n"
                      << "Average : " << stat.average << "ms,\n"
                      << "Stdev   : " << stat.stdev << "ms,\n"
                      << "95%     : " << stat.percentile_95 << "ms,\n"
                      << "99%     : " << stat.percentile_99 << "ms,\n"
                      << "99.9%   : " << stat.percentile_999 << "ms,\n";
        }
        {
            auto stat = get_statistics(durations_erase);
            std::cout << "Time to erase " << max_n << " elements:\n"
                      << "Average : " << stat.average << "ms,\n"
                      << "Stdev   : " << stat.stdev << "ms,\n"
                      << "95%     : " << stat.percentile_95 << "ms,\n"
                      << "99%     : " << stat.percentile_99 << "ms,\n"
                      << "99.9%   : " << stat.percentile_999 << "ms,\n";
        }
    }
}

int main() {
    namespace fc = frozenca;

    std::cout << "Balanced tree test\n";
    {
        fc::BTreeSet<std::string, 64> btree;
        // warm up for benchmark
        tree_perf_test(btree, true);
    }
    std::cout << "Warming up complete...\n";

    {
        std::cout << "frozenca::BTreeSet test (fanout 64 - default)\n";
        fc::BTreeSet<std::string, 64> btree;
        tree_perf_test(btree);
    }
    {
        std::cout << "frozenca::BTreeSet test (fanout 128)\n";
        fc::BTreeSet<std::string, 128> btree;
        tree_perf_test(btree);
    }
    {
        std::cout << "std::set test\n";
        std::set<std::string> rbtree;
        tree_perf_test(rbtree);
    }
}