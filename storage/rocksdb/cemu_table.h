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
#include <memory>
#include <mutex>
#include <string>

#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "table/multiget_context.h"
#include "table/table_reader.h"

// CemuTableReader wraps any TableReader and offloads MVCC filtering to the
// CEMU emulated CSD for user-facing forward scans. All other operations
// delegate to inner_ unchanged.
//
// On NewIterator() for a user scan with a snapshot, it:
//   1. Downloads and activates mvcc_filter.so on the CSD (once per instance).
//   2. Creates a Memory Range Set mapping the SST data block region (input)
//      and an FDM output buffer.
//   3. Executes the CSF synchronously via ioctl.
//   4. Reads the flat KV stream back from the FDM buffer.
//   5. Returns a CemuResultIterator over the filtered results.
//
// The guard (caller != kUserIterator || snapshot == nullptr) ensures that
// compaction, flush, and point-lookup paths always use the unfiltered reader.
class CemuTableReader : public rocksdb::TableReader {
 public:
  CemuTableReader(std::unique_ptr<rocksdb::TableReader> inner,
                  std::string file_path, uint64_t file_number)
      : inner_(std::move(inner)),
        file_path_(std::move(file_path)),
        file_number_(file_number) {}

  ~CemuTableReader() override;

  rocksdb::InternalIterator *NewIterator(
      const rocksdb::ReadOptions &read_options,
      const rocksdb::SliceTransform *prefix_extractor, rocksdb::Arena *arena,
      bool skip_filters, rocksdb::TableReaderCaller caller,
      size_t compaction_readahead_size = 0,
      bool allow_unprepared_value = false) override;

  // All delegation methods below forward to inner_ unchanged.

  rocksdb::FragmentedRangeTombstoneIterator *NewRangeTombstoneIterator(
      const rocksdb::ReadOptions &read_options) override {
    return inner_->NewRangeTombstoneIterator(read_options);
  }

  rocksdb::FragmentedRangeTombstoneIterator *NewRangeTombstoneIterator(
      rocksdb::SequenceNumber read_seqno,
      const rocksdb::Slice *timestamp) override {
    return inner_->NewRangeTombstoneIterator(read_seqno, timestamp);
  }

  uint64_t ApproximateOffsetOf(const rocksdb::ReadOptions &read_options,
                               const rocksdb::Slice &key,
                               rocksdb::TableReaderCaller caller) override {
    return inner_->ApproximateOffsetOf(read_options, key, caller);
  }

  uint64_t ApproximateSize(const rocksdb::ReadOptions &read_options,
                           const rocksdb::Slice &start,
                           const rocksdb::Slice &end,
                           rocksdb::TableReaderCaller caller) override {
    return inner_->ApproximateSize(read_options, start, end, caller);
  }

  void SetupForCompaction() override { inner_->SetupForCompaction(); }

  std::shared_ptr<const rocksdb::TableProperties> GetTableProperties()
      const override {
    return inner_->GetTableProperties();
  }

  size_t ApproximateMemoryUsage() const override {
    return inner_->ApproximateMemoryUsage();
  }

  rocksdb::Status Get(const rocksdb::ReadOptions &read_options,
                      const rocksdb::Slice &key,
                      rocksdb::GetContext *get_context,
                      const rocksdb::SliceTransform *prefix_extractor,
                      bool skip_filters = false) override {
    return inner_->Get(read_options, key, get_context, prefix_extractor,
                       skip_filters);
  }

  void MultiGet(const rocksdb::ReadOptions &read_options,
                const rocksdb::MultiGetContext::Range *mget_range,
                const rocksdb::SliceTransform *prefix_extractor,
                bool skip_filters = false) override {
    inner_->MultiGet(read_options, mget_range, prefix_extractor, skip_filters);
  }

  rocksdb::Status Prefetch(const rocksdb::ReadOptions &read_options,
                           const rocksdb::Slice *begin = nullptr,
                           const rocksdb::Slice *end = nullptr) override {
    return inner_->Prefetch(read_options, begin, end);
  }

  rocksdb::Status VerifyChecksum(const rocksdb::ReadOptions &read_options,
                                 rocksdb::TableReaderCaller caller) override {
    return inner_->VerifyChecksum(read_options, caller);
  }

  void Prepare(const rocksdb::Slice &target) override {
    inner_->Prepare(target);
  }

 private:
  // Loads mvcc_filter.so onto the CSD and activates it. No-op if already done.
  // Returns false and falls back to inner_ iterator on any CEMU setup error.
  bool EnsureCsfLoaded();

  std::unique_ptr<rocksdb::TableReader> inner_;
  std::string file_path_;
  uint64_t file_number_;

  // CEMU state — initialised lazily by EnsureCsfLoaded().
  int cemu_fd_ = -1;  // fd for /dev/nvme0c3
  int pind_ = -1;     // program index after CSF activation
  bool csf_ready_ = false;
  mutable std::once_flag csf_once_;
};

// CemuTableFactory wraps any TableFactory and replaces the created TableReader
// with a CemuTableReader. NewTableBuilder() is delegated unchanged so that
// compaction and flush write paths are unaffected.
class CemuTableFactory : public rocksdb::TableFactory {
 public:
  explicit CemuTableFactory(std::shared_ptr<rocksdb::TableFactory> inner)
      : inner_(std::move(inner)) {}

  const char *Name() const override { return "CemuTableFactory"; }

  rocksdb::Status NewTableReader(
      const rocksdb::ReadOptions &ro,
      const rocksdb::TableReaderOptions &table_reader_options,
      std::unique_ptr<rocksdb::RandomAccessFileReader> &&file,
      uint64_t file_size, std::unique_ptr<rocksdb::TableReader> *table_reader,
      bool prefetch_index_and_filter_in_cache) const override;

  rocksdb::TableBuilder *NewTableBuilder(
      const rocksdb::TableBuilderOptions &table_builder_options,
      rocksdb::WritableFileWriter *file) const override {
    return inner_->NewTableBuilder(table_builder_options, file);
  }

  bool IsDeleteRangeSupported() const override {
    return inner_->IsDeleteRangeSupported();
  }

 private:
  std::shared_ptr<rocksdb::TableFactory> inner_;
};
