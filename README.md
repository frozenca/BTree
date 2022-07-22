# B-Tree
This library implements a general-purpose header-only STL-like B-Tree in C++, including supports for using it for memory-mapped disk files and fixed-size allocators.

A B-Tree is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time. Unlike other self-balancing binary search trees, the B-tree is well suited for storage systems that read and write relatively large blocks of data, such as databases and file systems. (Wikipedia: https://en.wikipedia.org/wiki/B-tree)

Just like ordered associative containers in the C++ standard library, key-value pairs can be supported and duplicates can be allowed.

There are four specialized B-Tree classes: ```frozenca::BTreeSet```, ```frozenca::BTreeMultiSet```, ```frozenca::BTreeMap``` and ```frozenca::BTreeMultiMap```, which corresponds to ```std::set```, ```std::multiset```, ```std::map``` and ```std::multimap``` respectively.

# How to use
This library is header-only, so no additional setup process is required beyond including the headers.

# Target OS/Compiler version
This library aggressively uses C++20 features, and verified to work in gcc 11.2 and MSVC 19.32.

POSIX and Windows operating systems are supported in order to use the memory-mapped disk file interface.

There are currently no plans to support C++17 and earlier.

# Example usages
Usage is very similar to the C++ standard library ordered associative containers (i.e. ```std::set``` and its friends)
```
#include "fc_btree.h"
#include <iostream>
#include <string>

int main() {
  namespace fc = frozenca;
  fc::BTreeSet<int> btree;
  
  btree.insert(3);
  btree.insert(4);
  btree.insert(2);
  btree.insert(1);
  btree.insert(5);
  
  // 1 2 3 4 5
  std::cout << btree << '\n';
  
  fc::BTreeMap<std::string, int> strtree;

  strtree["asd"] = 3;
  strtree["a"] = 6;
  strtree["bbb"] = 9;
  strtree["asdf"] = 8;
  
  std::cout << strtree << '\n';

  strtree["asdf"] = 333;
  
  // 333
  std::cout << strtree["asdf"] << '\n';

  strtree.emplace("asdfgh", 200);
  std::cout << btree << '\n';
}
```

You can refer more example usages in ```test/unittest.cpp```.

Users can specify a fanout parameter for B-tree.

The default value is 2, where a B-Tree boils down to an 2-3-4 tree: (https://en.wikipedia.org/wiki/2%E2%80%933%E2%80%934_tree)

It is recommended to users to choose a fanout parameter suitable to their usages, instead of using the default value.

```
  // btree with fanout 4
  fc::BTreeSet<int, 4> btree;
```

# Supported operations
Other than regular operations supported by ```std::set``` and its friends (```lower_bound()```, ```upper_bound()```, ```equal_range()``` and etc), the following operations are supported:

```tree.kth(std::ptrdiff_t k)``` : Returns the k-th element in the tree as 0-based index. Time complexity: ```O(log n)```

```tree.order(const_iterator_type iter)``` : Returns the rank of the element in the iterator in the tree as 0-based index. Time complexity: ```O(log n)```

```tree.enumerate(const key_type& a, const key_type& b)``` : Range query. Returns the range of values for their key in ```[a, b]```. Time complexity: ```O(log n + k)``` where ```k``` is the number of keys in the range

```frozenca::join(Tree&& tree1, value_type val, Tree&& tree2)``` : Joins two trees to a single tree. The largest key in ```tree1``` should be less than or equal to the key of ```val``` and the smallest key in ```tree2``` should be greater than or equal to the key of ```val```. Time complexity: ```O(1 + diff_height)```

```frozenca::split(Tree&& tree, value_type val)``` : Splits a tree to two trees, so that the first tree contains keys less than the key of ```val```, and the second tree contains keys greater than the key of ```val```. Time complexity: ```O(log n)```

# Concurrency
Currently, thread safety is not guaranteed. Lock-free support is the first TODO, but contributions are welcome if you're interested.

# Disk B-Tree
You can use a specialized variant that utilizes memory-mapped disk files and an associated fixed-size allocator. You have to include ```fc_disk_btree.h```, ```fc_disk_fixed_alloc.h``` and ```fc_mmfile.h``` to use it.

For this variant, key-value pairs are supported, but duplicates are not supported (but you can easily adjust it to support it manually).

For this variant, supported types have stricter type constraints: it should satisfy ```std::trivially_copyable_v```, and its alignment should at least be the alignment of the pointer type in the machine (for both key type and value type for key-value pairs). This variant has a larger default fanout, 64.

The following code initializes a ```frozenca::DiskBTree```, which generates a memory-mapped disk file ```database.bin``` and uses it, with an initial byte size of 32 megabytes. If the third argument is ```true```, it will destroy the existing file and create a new one (default is ```false```). You can't extend the pool size of the memory-mapped disk file once you initialized (doing so invalidates all pointers in the associated allocator).

```
fc::DiskBTree<std::int64_t, 128> btree("database.bin", 1UL << 25UL, true);
```

# Performance
Using a performance test code (```test/perftest.cpp```) that insert/retrieve/erase 10 million ``std::int64_t`` in random order, I see the following results in my machine (gcc 11.2-O3):

```
Balanced tree test
Warming up complete...
frozenca::DiskBTreeSet test
Time to insert 10000000 elements: 5165.81 ms
Time to lookup 10000000 elements: 4784.74 ms
Time to erase 10000000 elements: 6180.8 ms
std::set test
Time to insert 10000000 elements: 12889.7 ms
Time to lookup 10000000 elements: 15527.5 ms
Time to erase 10000000 elements: 18128.5 ms
```

# License
This library is licensed under Apache License Version 2.0 with LLVM Exceptions (LICENSE-Apache2-LLVM or https://llvm.org/foundation/relicensing/LICENSE.txt)





