/*
   Copyright (c) 2026, Percona and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>

#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/internal_iterator.h"

namespace myrocks {
extern std::atomic<uint64_t> rocksdb_nvmevirt_keys_seen;
extern std::atomic<uint64_t> rocksdb_nvmevirt_keys_filtered;
}  // namespace myrocks

// NvmeVirtResultIterator walks the flat KV stream produced by the
// __rocksdb_mvcc_filter CSD kernel and read back via
// CSDVirt::csdvirt_read_slm() into host memory.
//
// Stream format:  [uint32 key_len][key bytes][uint32 value_len][value bytes] ...
// (matches struct rocksdb_mvcc_filter_output's header format in
// csd_user_func.h -- the header itself is stripped by the caller
// before constructing this iterator; buf/len here cover only the KV stream).
//
// The buffer is already MVCC-filtered WITHIN this one SST file: every surviving
// entry is visible to the snapshot and is the newest visible version of its
// user_key found in this file. It intentionally still contains any live
// tombstones (kTypeDeletion/kTypeSingleDeletion) unfiltered -- those must reach
// DBIter so its own (unmodified) cross-file merge logic can shadow older, still
// -visible versions of the same key that live in a different, older SST.
//
// Forward-only: Prev()/SeekToLast()/SeekForPrev() are not implemented because
// the CSD kernel scans the SST in forward order and writes results
// sequentially. Callers that require reverse iteration must use the
// unfiltered inner reader.
class NvmeVirtResultIterator : public rocksdb::InternalIterator {
 public:
  // Takes ownership of buf (heap-allocated, must be freed with delete[]).
  // keys_seen / keys_filtered come from the rocksdb_mvcc_filter_output header
  // the caller already parsed and stripped off buf/len.
  NvmeVirtResultIterator(char *buf, size_t len, uint64_t keys_seen,
                         uint64_t keys_filtered)
      : buf_(buf),
        len_(len),
        keys_seen_(keys_seen),
        keys_filtered_(keys_filtered) {
    SeekToFirst();
  }

  ~NvmeVirtResultIterator() override {
    myrocks::rocksdb_nvmevirt_keys_seen.fetch_add(keys_seen_,
                                                   std::memory_order_relaxed);
    myrocks::rocksdb_nvmevirt_keys_filtered.fetch_add(
        keys_filtered_, std::memory_order_relaxed);
    delete[] buf_;
  }

  bool Valid() const override { return valid_; }

  void SeekToFirst() override {
    pos_ = 0;
    valid_ = false;
    parse_current();
  }

  void SeekToLast() override {
    // Not supported for batch-forward output.
    valid_ = false;
  }

  void Seek(const rocksdb::Slice &target) override {
    // Linear scan from the beginning. Output is sorted (SST data blocks are
    // sorted), so we stop as soon as current key >= target.
    SeekToFirst();
    while (valid_ && key_.compare(target) < 0) Next();
  }

  void SeekForPrev(const rocksdb::Slice & /*target*/) override {
    valid_ = false;
  }

  void Next() override {
    if (!valid_) return;
    pos_ = next_pos_;
    parse_current();
  }

  void Prev() override { valid_ = false; }

  rocksdb::Slice key() const override { return key_; }
  rocksdb::Slice value() const override { return value_; }
  rocksdb::Status status() const override { return rocksdb::Status::OK(); }

  bool IsKeyPinned() const override { return true; }
  bool IsValuePinned() const override { return true; }

 private:
  void parse_current() {
    if (pos_ + 4 > len_) {
      valid_ = false;
      return;
    }

    uint32_t klen;
    memcpy(&klen, buf_ + pos_, 4);
    const size_t after_klen = pos_ + 4;
    if (after_klen + klen + 4 > len_) {
      valid_ = false;
      return;
    }

    uint32_t vlen;
    memcpy(&vlen, buf_ + after_klen + klen, 4);
    const size_t after_vlen = after_klen + klen + 4;
    if (after_vlen + vlen > len_) {
      valid_ = false;
      return;
    }

    key_ = rocksdb::Slice(buf_ + after_klen, klen);
    value_ = rocksdb::Slice(buf_ + after_vlen, vlen);
    next_pos_ = after_vlen + vlen;
    valid_ = true;
  }

  char *buf_;
  size_t len_;
  size_t pos_ = 0;
  size_t next_pos_ = 0;
  bool valid_ = false;
  rocksdb::Slice key_;
  rocksdb::Slice value_;

  uint64_t keys_seen_;
  uint64_t keys_filtered_;
};
