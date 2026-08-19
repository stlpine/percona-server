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
#include <string>

#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "table/multiget_context.h"
#include "table/table_builder.h"
#include "table/table_reader.h"

class THD;

namespace myrocks {
// Plain bool, not atomic -- the convention already used throughout this
// codebase for GLOBAL sysvars: MySQL's
// SET GLOBAL plumbing writes it directly, other threads read it directly, with
// no stronger synchronization than a MySQL system variable normally gets.
extern bool rocksdb_nvmevirt_enabled;

// The offload loads a whole SST file into a single SLM allocation cycle with
// no adaptive/demand loading, so anything bigger falls back to the
// unfiltered reader. A GLOBAL sysvar, not a compile-time constant, because
// the right value is deployment-dependent on nvmev.ko's own `slm_size=`
// insmod parameter (CSDVirt.hpp has no runtime query API for it): the input
// buffer holds ~1x the file size and the output buffer ~2x, both live at
// once under g_nvmevirt_exec_mutex, so this should be budgeted at roughly
// slm_size / 3 with margin, biased to a power of two to avoid the SLM
// buddy allocator rounding up on allocation.
extern unsigned long long  // NOLINT(runtime/int)
    rocksdb_nvmevirt_max_sst_bytes;

// Caller-restriction check: true iff `thd`'s session has
// `SET [SESSION] rocksdb_nvmevirt_olap_session = 1`. Defined in ha_rocksdb.cc
// (the THDVAR(thd, name) macro only resolves within the translation unit
// that declared the THDVAR via MYSQL_THDVAR_BOOL, so this file can't read it
// directly). rocksdb_nvmevirt_enabled above is still the GLOBAL master
// switch; this narrows eligibility further to sessions that opt in, so a
// benchmark harness's OLAP connection can enable offload for itself without
// pulling concurrent OLTP connections onto the same offload path/mutex.
bool rdb_nvmevirt_olap_session(THD *thd);
}  // namespace myrocks

// NvmeVirtTableReader wraps any TableReader and offloads MVCC filtering to the
// NVMeVirt emulated CSD for user-facing forward scans, for
// `nvmevirt_`-prefixed column families. This is a NAIVE v1 baseline:
// filtering happens ONE SST FILE AT A TIME, at the TableReader layer --
// cross-file/cross-level MVCC correctness is still entirely the job of
// RocksDB's own unmodified MergingIterator + DBIter::FindNextUserEntry,
// unchanged by this feature.
//
// On NewIterator() for a user scan with a snapshot, it:
//   1. Ensures the process-wide CSDVirt device handle is open.
//   2. Allocates an input SLM buffer sized to the whole SST file and loads the
//      file into it via CSDVirt::csdvirt_load_files() (the whole file, not
//      just the data-block region -- the kernel parses the footer/metaindex/
//      index block itself, see the device-side __rocksdb_mvcc_filter()).
//   3. Allocates an output SLM buffer and runs
//      ROCKSDB_MVCC_FILTER_PROGRAM_INDEX synchronously via
//      CSDVirt::csdvirt_execute() (no async/dependency-table machinery used
//      in v1).
//   4. Reads the filtered flat KV stream back via CSDVirt::csdvirt_read_slm().
//   5. Returns a NvmeVirtResultIterator over the filtered results.
//
// The guard (caller != kUserIterator || snapshot == nullptr ||
// !rocksdb_nvmevirt_enabled || !rdb_nvmevirt_olap_session(current_thd))
// ensures that compaction, flush, and point-lookup paths always use the
// unfiltered reader -- and, as of v2, that only sessions which have
// explicitly opted in via `SET SESSION rocksdb_nvmevirt_olap_session = 1`
// reach the offload path at all. This is the OLAP/OLTP caller-restriction
// fix: previously every kUserIterator caller with a snapshot qualified,
// which meant concurrent OLTP range-read iterators on the same CF competed
// for g_nvmevirt_exec_mutex on equal footing with the OLAP scan itself. Any
// CSD-side error, or an SST larger
// than rocksdb_nvmevirt_max_sst_bytes (no adaptive/demand loading for inputs
// that don't fit in one SLM allocation), falls back to the unfiltered inner
// iterator -- the offload is always best-effort, never required for
// correctness.
class NvmeVirtTableReader : public rocksdb::TableReader {
 public:
  NvmeVirtTableReader(std::unique_ptr<rocksdb::TableReader> inner,
                       std::string file_path, uint64_t file_size)
      : inner_(std::move(inner)),
        file_path_(std::move(file_path)),
        file_size_(file_size) {}

  ~NvmeVirtTableReader() override = default;

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
  // Runs the full alloc/load/execute/read_slm/release sequence against the CSD
  // for this SST file. On success, *out_buf is a heap-allocated buffer (caller
  // takes ownership, must delete[] it) holding ONLY the flat KV stream (the
  // rocksdb_mvcc_filter_output header has already been parsed out into
  // *keys_seen/*keys_filtered and stripped from *out_buf), and *out_len is its
  // length in bytes.
  rocksdb::Status RunMvccFilter(uint64_t snapshot_seq, char **out_buf,
                                size_t *out_len, uint64_t *keys_seen,
                                uint64_t *keys_filtered) const;

  std::unique_ptr<rocksdb::TableReader> inner_;
  std::string file_path_;
  uint64_t file_size_;
};

// NvmeVirtTableFactory wraps any TableFactory and replaces the created
// TableReader with a NvmeVirtTableReader. NewTableBuilder() is delegated
// unchanged so that compaction and flush write paths are unaffected -- this
// feature only ever reads existing, ordinary SSTs; it never changes what gets
// written.
class NvmeVirtTableFactory : public rocksdb::TableFactory {
 public:
  explicit NvmeVirtTableFactory(std::shared_ptr<rocksdb::TableFactory> inner)
      : inner_(std::move(inner)) {}

  const char *Name() const override { return "NvmeVirtTableFactory"; }

  // Base class has two overloads of NewTableReader -- an older 5-param one
  // (non-pure, defaults to a fresh ReadOptions and delegates to the 6-param
  // one) and the real pure-virtual 6-param one below, which we override.
  // Declaring NewTableReader at all hides the 5-param overload from this
  // derived class unless explicitly pulled back in here.
  using rocksdb::TableFactory::NewTableReader;

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
