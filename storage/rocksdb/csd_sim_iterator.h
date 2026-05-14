#pragma once

#include <atomic>
#include <string>

#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/internal_iterator.h"
#include "util/coding.h"

// Global counters updated by CsdSimIterator destructors.
// Exposed as SHOW STATUS variables rocksdb_csd_sim_keys_seen and
// rocksdb_csd_sim_keys_filtered.  Defined in ha_rocksdb.cc inside namespace
// myrocks — the extern must match that namespace to avoid a linker mismatch
// that would silently leave the status variables at zero.
namespace myrocks {
extern std::atomic<uint64_t> rocksdb_csd_sim_keys_seen;
extern std::atomic<uint64_t> rocksdb_csd_sim_keys_filtered;
}  // namespace myrocks

// CsdSimIterator wraps a BlockBasedTableIterator and simulates MVCC filter
// pushdown to a Computational Storage Device.
//
// It applies two filter rules before returning an entry to DBIter:
//   1. seq > snapshot_seq → invisible (after snapshot), skip.
//   2. user_key == prev_user_key → older version of same key, skip.
//      (keys in a data block are sorted by user_key ASC, seq DESC, so
//       the first visible entry per user_key is the canonical version)
//
// kTypeMerge entries pass through unfiltered; DBIter accumulates operands.
// Prev() / SeekToLast() are unfiltered (reverse scan support deferred).
class CsdSimIterator : public rocksdb::InternalIterator {
 public:
  CsdSimIterator(rocksdb::InternalIterator *inner,
                 rocksdb::SequenceNumber snapshot_seq)
      : inner_(inner), snapshot_seq_(snapshot_seq) {}

  ~CsdSimIterator() override {
    myrocks::rocksdb_csd_sim_keys_seen.fetch_add(keys_seen_,
                                                  std::memory_order_relaxed);
    myrocks::rocksdb_csd_sim_keys_filtered.fetch_add(keys_filtered_,
                                                      std::memory_order_relaxed);
    // inner_ was created with arena=nullptr (forced in CsdSimTableReader) so
    // it is always heap-allocated and safe to delete here.
    delete inner_;
  }

  bool Valid() const override { return inner_->Valid(); }

  void SeekToFirst() override {
    prev_user_key_.clear();
    inner_->SeekToFirst();
    FindNextEmittable();
  }

  void SeekToLast() override { inner_->SeekToLast(); }

  void Seek(const rocksdb::Slice &target) override {
    prev_user_key_.clear();
    inner_->Seek(target);
    FindNextEmittable();
  }

  void SeekForPrev(const rocksdb::Slice &target) override {
    inner_->SeekForPrev(target);
  }

  void Next() override {
    inner_->Next();
    FindNextEmittable();
  }

  void Prev() override { inner_->Prev(); }

  rocksdb::Slice key() const override { return inner_->key(); }
  rocksdb::Slice value() const override { return inner_->value(); }
  rocksdb::Status status() const override { return inner_->status(); }

  bool IsKeyPinned() const override { return inner_->IsKeyPinned(); }
  bool IsValuePinned() const override { return inner_->IsValuePinned(); }

  uint64_t keys_seen() const { return keys_seen_; }
  uint64_t keys_filtered() const { return keys_filtered_; }

 private:
  // Advances inner_ until we find an entry to emit, or inner_ becomes invalid.
  // Updates prev_user_key_ when an entry is accepted.
  void FindNextEmittable() {
    while (inner_->Valid()) {
      rocksdb::Slice ikey = inner_->key();
      if (ikey.size() < 8) break;  // malformed key — stop and let DBIter handle

      // Internal key footer: last 8 bytes = (seq << 8) | type
      uint64_t packed =
          rocksdb::DecodeFixed64(ikey.data() + ikey.size() - 8);
      uint64_t seq = packed >> 8;
      uint8_t type = static_cast<uint8_t>(packed & 0xff);

      keys_seen_++;

      // Rule 1: invisible — sequence number is newer than the snapshot.
      if (seq > snapshot_seq_) {
        keys_filtered_++;
        inner_->Next();
        continue;
      }

      // User key is all bytes except the 8-byte footer.
      rocksdb::Slice user_key(ikey.data(), ikey.size() - 8);

      // Rule 2: older version of the same user_key — already emitted a newer
      // visible version for this key in this scan.
      if (user_key == prev_user_key_) {
        keys_filtered_++;
        inner_->Next();
        continue;
      }

      // kTypeMerge (0x2): pass through without updating prev_user_key_ so
      // subsequent merge operands for the same key are also passed through.
      if (type == 0x2 /* kTypeMerge */) {
        return;
      }

      // kTypeValue (0x1) or kTypeDeletion (0x0): latest visible version.
      prev_user_key_.assign(user_key.data(), user_key.size());
      return;
    }
  }

  rocksdb::InternalIterator *inner_;
  rocksdb::SequenceNumber snapshot_seq_;
  std::string prev_user_key_;
  uint64_t keys_seen_{0};
  uint64_t keys_filtered_{0};
};
