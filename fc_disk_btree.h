#ifndef __FC_DISK_BTREE_H__
#define __FC_DISK_BTREE_H__

#include "fc_btree.h"
#include "fc_disk_fixed_alloc.h"

namespace frozenca {

template <DiskAllocable K, index_t t = 64, typename Comp = std::ranges::less>
class DiskBTree
    : public BTreeBase<K, K, t, t, Comp, false, AllocatorFixed<K, 2 * t - 1>> {
private:
  MemoryMappedFile mm_file_;
  MemoryResourceFixed<K, 2 * t - 1> mem_res_;

public:
  using Base = BTreeBase<K, K, t, t, Comp, false, AllocatorFixed<K, 2 * t - 1>>;

  DiskBTree(const MemoryMappedFile &mm_file)
      : mm_file_{mm_file}, mem_res_{reinterpret_cast<unsigned char *>(
                                        mm_file_.data()),
                                    mm_file_.size()} {
    Base(AllocatorFixed<K, 2 * t - 1>(&mem_res_));
  }

  DiskBTree(const std::filesystem::path &path, std::size_t pool_size,
            bool trunc = false)
      : mm_file_{path, pool_size, trunc}, mem_res_{
                                              reinterpret_cast<unsigned char *>(
                                                  mm_file_.data()),
                                              mm_file_.size()} {
    Base(AllocatorFixed<K, 2 * t - 1>(&mem_res_));
  }
};

template <DiskAllocable K, DiskAllocable V, index_t t = 64,
          typename Comp = std::ranges::less>
class DiskBTreeMap
    : public BTreeBase<K, DiskPair<K, V>, t, t, Comp, false,
                       AllocatorFixed<DiskPair<K, V>, 2 * t - 1>> {
private:
  MemoryMappedFile mm_file_;
  MemoryResourceFixed<DiskPair<K, V>, 2 * t - 1> mem_res_;

  static_assert(DiskAllocable<DiskPair<K, V>>);

public:
  using Base = BTreeBase<K, DiskPair<K, V>, t, t, Comp, false,
                         AllocatorFixed<DiskPair<K, V>, 2 * t - 1>>;

  DiskBTreeMap(const MemoryMappedFile &mm_file)
      : mm_file_{mm_file}, mem_res_{reinterpret_cast<unsigned char *>(
                                        mm_file_.data()),
                                    mm_file_.size()} {
    Base(AllocatorFixed<DiskPair<K, V>, 2 * t - 1>(&mem_res_));
  }

  DiskBTreeMap(const std::filesystem::path &path, std::size_t pool_size,
               bool trunc = false)
      : mm_file_{path, pool_size, trunc}, mem_res_{
                                              reinterpret_cast<unsigned char *>(
                                                  mm_file_.data()),
                                              mm_file_.size()} {
    Base(AllocatorFixed<DiskPair<K, V>, 2 * t - 1>(&mem_res_));
  }
};

} // namespace frozenca

#endif //__FC_DISK_BTREE_H__
