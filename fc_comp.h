#ifndef __FC_COMP_H__
#define __FC_COMP_H__

#include <bit>
#include <bitset>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <immintrin.h>
#include <type_traits>

namespace frozenca {

template <typename K>
concept CanUseSimd = (sizeof(K) == 4 || sizeof(K) == 8) && (std::signed_integral<K> || std::floating_point<K>);

using regi = __m512i;
using regf = __m512;
using regd = __m512d;

unsigned int cmp(std::int32_t key, const std::int32_t* key_ptr) {
    regi key_broadcasted = _mm512_set1_epi32(key);
    regi keys_to_comp = _mm512_load_si512(reinterpret_cast<const regi*>(key_ptr));
    return _mm512_cmpgt_epi32_mask(key_broadcasted, keys_to_comp);
}

unsigned int cmp(std::int64_t key, const std::int64_t* key_ptr) {
    regi key_broadcasted = _mm512_set1_epi64(key);
    regi keys_to_comp = _mm512_load_si512(reinterpret_cast<const regi*>(key_ptr));
    return _mm512_cmpgt_epi64_mask(key_broadcasted, keys_to_comp);
}

unsigned int cmp(float key, const float* key_ptr) {
    regf key_broadcasted = _mm512_set1_ps(key);
    regf keys_to_comp = _mm512_load_ps(key_ptr);
    return _mm512_cmp_ps_mask(key_broadcasted, keys_to_comp, _MM_CMPINT_GT);
}

unsigned int cmp(double key, const double* key_ptr) {
    regd key_broadcasted = _mm512_set1_pd(key);
    regd keys_to_comp = _mm512_load_pd(key_ptr);
    return _mm512_cmp_pd_mask(key_broadcasted, keys_to_comp, _MM_CMPINT_GT);
}

unsigned int cmp(const std::int32_t* key_ptr, std::int32_t key) {
    regi key_broadcasted = _mm512_set1_epi32(key);
    regi keys_to_comp = _mm512_load_si512(reinterpret_cast<const regi*>(key_ptr));
    return _mm512_cmpgt_epi32_mask(keys_to_comp, key_broadcasted);
}

unsigned int cmp(const std::int64_t* key_ptr, std::int64_t key) {
    regi key_broadcasted = _mm512_set1_epi64(key);
    regi keys_to_comp = _mm512_load_si512(reinterpret_cast<const regi*>(key_ptr));
    return _mm512_cmpgt_epi64_mask(keys_to_comp, key_broadcasted);
}

unsigned int cmp(const float* key_ptr, float key) {
    regf key_broadcasted = _mm512_set1_ps(key);
    regf keys_to_comp = _mm512_load_ps(key_ptr);
    return _mm512_cmp_ps_mask(keys_to_comp, key_broadcasted, _MM_CMPINT_GT);
}

unsigned int cmp(const double* key_ptr, double key) {
    regd key_broadcasted = _mm512_set1_pd(key);
    regd keys_to_comp = _mm512_load_pd(key_ptr);
    return _mm512_cmp_pd_mask(keys_to_comp, key_broadcasted, _MM_CMPINT_GT);
}

template <CanUseSimd K>
struct SimdTrait {
    static constexpr int shift = (sizeof(K) == 4) ? 4 : 3;
    static constexpr int mask = (sizeof(K) == 4) ? 0xF : 0x7;
    static constexpr int unit = (sizeof(K) == 4) ? 16 : 8;
};

template <CanUseSimd K, bool less>
inline std::int32_t get_lb_simd(K key, const K* first, const K* last) {
    auto len = static_cast<std::int32_t>(last - first);
    // make to the least multiple of SimdUnit which is at least len
    len = ((len >> SimdTrait<K>::shift) + ((len & SimdTrait<K>::mask) ? 1 : 0)) << SimdTrait<K>::shift;

    const K* curr = first;
    std::int32_t i = 0;
    int mask = 0;
    auto half = (len >> (SimdTrait<K>::shift + 1)) << SimdTrait<K>::shift;
    while (len > SimdTrait<K>::unit) {
        len -= half;
        auto next_half = (len >> (SimdTrait<K>::shift + 1)) << SimdTrait<K>::shift;
        __builtin_prefetch(curr + next_half - SimdTrait<K>::unit);
        __builtin_prefetch(curr + half + next_half - SimdTrait<K>::unit);
        auto mid = curr + half - SimdTrait<K>::unit;
        if constexpr (less) {
            mask = ~cmp(key, mid);
        } else {
            mask = ~cmp(mid, key);
        }
        i = __builtin_ffs(mask) - 1;
        curr += (i == SimdTrait<K>::unit) * half;
        if (i & SimdTrait<K>::mask) {
            return static_cast<std::int32_t>(mid - first) + i;
        }
        half = next_half;
    }
    if constexpr (less) {
        mask = ~cmp(key, curr);
    } else {
        mask = ~cmp(curr, key);
    }
    i = __builtin_ffs(mask) - 1;
    return std::min(static_cast<std::int32_t>(last - first), static_cast<std::int32_t>(curr - first) + i);
}

template <CanUseSimd K, bool less>
inline std::int32_t get_ub_simd(K key, const K* first, const K* last) {
    auto len = static_cast<std::int32_t>(last - first);
    // make to the least multiple of SimdUnit which is at least len
    len = ((len >> SimdTrait<K>::shift) + ((len & SimdTrait<K>::mask) ? 1 : 0)) << SimdTrait<K>::shift;

    const K* curr = first;
    std::int32_t i = 0;
    int mask = 0;
    while (len > SimdTrait<K>::unit) {
        auto half = (len >> (SimdTrait<K>::shift + 1)) << SimdTrait<K>::shift;
        len -= half;
        auto mid = curr + half - SimdTrait<K>::unit;
        if constexpr (less) {
            mask = cmp(mid, key);
        } else {
            mask = cmp(key, mid);
        }
        i = __builtin_ffs(mask) - 1;
        curr += (mask == 0) * half;
        if (i > 0) {
            return static_cast<std::int32_t>(mid - first) + i;
        }
    }
    if constexpr (less) {
        mask = cmp(curr, key);
    } else {
        mask = cmp(key, curr);
    }
    i = (mask == 0) ? len : __builtin_ffs(mask) - 1;
    return std::min(static_cast<std::int32_t>(last - first), static_cast<std::int32_t>(curr - first) + i);
}

} // namespace frozenca

#endif //__FC_COMP_H__
