#pragma once

#include <memory>

#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "table/multiget_context.h"
#include "table/table_reader.h"

// CsdSimTableReader wraps any TableReader and injects CsdSimIterator for
// user-facing forward scans that carry an explicit snapshot. All other
// operations (Get, compaction reads, range tombstones, etc.) are delegated
// to the inner reader unchanged.
class CsdSimTableReader : public rocksdb::TableReader {
 public:
  explicit CsdSimTableReader(std::unique_ptr<rocksdb::TableReader> inner)
      : inner_(std::move(inner)) {}

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

  rocksdb::Status VerifyChecksum(
      const rocksdb::ReadOptions &read_options,
      rocksdb::TableReaderCaller caller) override {
    return inner_->VerifyChecksum(read_options, caller);
  }

  void Prepare(const rocksdb::Slice &target) override {
    inner_->Prepare(target);
  }

 private:
  std::unique_ptr<rocksdb::TableReader> inner_;
};

// CsdSimTableFactory wraps any TableFactory (typically BlockBasedTableFactory)
// and replaces the created TableReader with a CsdSimTableReader.
class CsdSimTableFactory : public rocksdb::TableFactory {
 public:
  explicit CsdSimTableFactory(
      std::shared_ptr<rocksdb::TableFactory> inner)
      : inner_(std::move(inner)) {}

  const char *Name() const override { return "CsdSimTableFactory"; }

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
