#include "./csd_sim_table.h"

#include "./csd_sim_iterator.h"

#include "rocksdb/table_reader_caller.h"

namespace myrocks {
extern bool rocksdb_csd_sim_enabled;
}  // namespace myrocks

rocksdb::InternalIterator *CsdSimTableReader::NewIterator(
    const rocksdb::ReadOptions &read_options,
    const rocksdb::SliceTransform *prefix_extractor, rocksdb::Arena *arena,
    bool skip_filters, rocksdb::TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  rocksdb::InternalIterator *iter = inner_->NewIterator(
      read_options, prefix_extractor, arena, skip_filters, caller,
      compaction_readahead_size, allow_unprepared_value);

  // Only wrap for user-facing forward scans with an explicit snapshot,
  // and only when the runtime toggle is on. Compaction, flush, and other
  // internal readers must always see all versions.
  if (caller != rocksdb::kUserIterator) return iter;
  if (read_options.snapshot == nullptr) return iter;
  if (!myrocks::rocksdb_csd_sim_enabled) return iter;

  rocksdb::SequenceNumber snap_seq =
      read_options.snapshot->GetSequenceNumber();
  return new CsdSimIterator(iter, snap_seq);
}

rocksdb::Status CsdSimTableFactory::NewTableReader(
    const rocksdb::ReadOptions &ro,
    const rocksdb::TableReaderOptions &table_reader_options,
    std::unique_ptr<rocksdb::RandomAccessFileReader> &&file, uint64_t file_size,
    std::unique_ptr<rocksdb::TableReader> *table_reader,
    bool prefetch_index_and_filter_in_cache) const {
  std::unique_ptr<rocksdb::TableReader> inner_reader;
  rocksdb::Status s = inner_->NewTableReader(
      ro, table_reader_options, std::move(file), file_size, &inner_reader,
      prefetch_index_and_filter_in_cache);
  if (!s.ok()) return s;
  *table_reader = std::make_unique<CsdSimTableReader>(std::move(inner_reader));
  return rocksdb::Status::OK();
}
