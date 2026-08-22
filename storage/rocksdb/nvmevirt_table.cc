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

#include "./nvmevirt_table.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>

#include "./nvmevirt_iterator.h"
#include "rocksdb/snapshot.h"
#include "rocksdb/status.h"
#include "sql/current_thd.h"
#include "sql/sql_class.h"

#ifdef HAVE_NVMEVIRT
// Host-side ioctl wrapper library for the NVMeVirt CSD kernel module
// (nvmev.ko).
#include "CSDVirt.hpp"
// CSD_PARAMS union, ROCKSDB_MVCC_FILTER_PROGRAM_INDEX, and struct
// rocksdb_mvcc_filter_{params,output}. This header is shared with the kernel
// module build but is written to be usable from userspace too, so the wire
// structs cannot drift between host and device.
#include "csd_user_func.h"
#endif  // HAVE_NVMEVIRT

// rocksdb_nvmevirt_enabled / rocksdb_nvmevirt_keys_seen / _keys_filtered are
// DEFINED in ha_rocksdb.cc (alongside their MYSQL_SYSVAR_BOOL/status-variable
// registration) -- see the `extern` declarations in nvmevirt_table.h /
// nvmevirt_iterator.h.

#ifdef HAVE_NVMEVIRT

namespace {

// Dedicated append-only debug log -- cheap visibility during bring-up
// without going through MySQL's own error log.
void nvmevirt_log(const char *fmt, ...) {
  FILE *f = fopen("/tmp/nvmevirt_debug.log", "a");
  if (f == nullptr) return;
  time_t now = time(nullptr);
  struct tm tm_buf;
  localtime_r(&now, &tm_buf);
  char ts[9];
  strftime(ts, sizeof(ts), "%H:%M:%S", &tm_buf);
  THD *thd = current_thd;
  fprintf(f, "[%s] thd=%lu ", ts, thd ? (unsigned long)thd->thread_id() : 0UL);
  va_list args;
  va_start(args, fmt);
  vfprintf(f, fmt, args);
  va_end(args);
  fclose(f);
}

// Process-wide CSDVirt handle. Calls need no serialising: the command struct is
// a per-call local, the fd map belongs to the unused csdvirt_open/load family,
// and the device side does its own locking. call_once still guards init.
CSDVirt *g_nvmevirt_csdvirt = nullptr;
std::once_flag g_nvmevirt_init_once;

void EnsureCsdvirtDevice() {
  std::call_once(g_nvmevirt_init_once, [] {
    g_nvmevirt_csdvirt = new CSDVirt();
    std::string device_path = get_csd_device_path();
    int rc = g_nvmevirt_csdvirt->csdvirt_init_dev(device_path.c_str());
    if (rc < 0) {
      nvmevirt_log(
          "EnsureCsdvirtDevice: csdvirt_init_dev(%s) failed (rc=%d) -- "
          "offload unavailable, all scans fall back to the unfiltered "
          "reader\n",
          device_path.c_str(), rc);
      delete g_nvmevirt_csdvirt;
      g_nvmevirt_csdvirt = nullptr;
    }
  });
}

}  // namespace

// FUTURE OPTIMIZATION (not done -- v1 scope is measuring baseline performance,
// not optimizing it): every call here does a full load+parse+filter cycle from
// scratch, even on repeat access to the same SST -- there's no host- or
// device-side cache of the parsed footer/metaindex/index chain keyed by SST
// identity. SSTs are immutable once written, so this is pure redundant work on
// repeat scans of the same file. See the matching note in the device-side
// __rocksdb_mvcc_filter scope comment for the other half of this.
rocksdb::Status NvmeVirtTableReader::RunMvccFilter(uint64_t snapshot_seq,
                                                    char **out_buf,
                                                    size_t *out_len,
                                                    uint64_t *keys_seen,
                                                    uint64_t *keys_filtered) const {
  // Per-call wall-clock breakdown: a cycles-based flamegraph cannot separate
  // host CPU from time blocked in the device round-trip, since a sleeping
  // thread produces no samples either way.
  using clock = std::chrono::steady_clock;
  auto t_start = clock::now();

  EnsureCsdvirtDevice();
  if (g_nvmevirt_csdvirt == nullptr) {
    return rocksdb::Status::Aborted("NvmeVirt CSD device unavailable");
  }

  CSDVirt *csdvirt = g_nvmevirt_csdvirt;

  // v1 sizes the output buffer conservatively at 2x the input file size.
  // Our output format replaces RocksDB's compact varint
  // shared/non_shared/value_length encoding with fixed 4-byte length
  // prefixes per key and per value -- which can be *larger* than the source
  // encoding for small, heavily-shared-prefix entries, even though we only
  // ever drop entries, never add them. Revisit this bound if "output buffer
  // full, truncating" warnings show up in nvmevirt_debug.log.
  const size_t output_capacity =
      sizeof(struct rocksdb_mvcc_filter_output) + 2 * file_size_;

  // The device's SLM allocator (alloc_slm_range) requires page-aligned
  // allocation sizes -- confirmed via a kernel BUG_ON crash ("Allocated SLM
  // offset is not aligned to page size") that took the whole emulated device
  // down when we passed raw file_size_/output_capacity directly. Its own
  // comment acknowledges the underlying buddy allocator doesn't guarantee
  // page alignment for arbitrary sizes and expects callers to pre-align.
  // kSlmPageSize must be a multiple of the device's SLM_PAGE_SIZE --
  // hardcoded here rather than shared via a common header, since that is a
  // kernel-module-only header (linux/kthread.h etc.) not includable from
  // this userspace plugin code. Only sizes handed to the device are rounded
  // up; file_size_ itself stays unaligned in sstable_size, since the
  // kernel-side parser needs the true byte count to locate the footer.
  constexpr size_t kSlmPageSize = 16 * 1024;
  auto align_up_to_slm_page = [](size_t size) {
    return (size + kSlmPageSize - 1) & ~(kSlmPageSize - 1);
  };

  // csdvirt_alloc_memory returns an SLM offset, and returns (size_t)-1 on
  // failure. Offset 0 is a valid allocation, the one a freshly loaded module
  // hands out first, so testing for 0 rejected a good address and made the
  // first offload of every session fall back to the host path.
  constexpr size_t kAllocFailed = static_cast<size_t>(-1);

  size_t input_addr =
      csdvirt->csdvirt_alloc_memory(align_up_to_slm_page(file_size_));
  if (input_addr == kAllocFailed) {
    return rocksdb::Status::Aborted("csdvirt_alloc_memory(input) failed");
  }
  size_t output_addr =
      csdvirt->csdvirt_alloc_memory(align_up_to_slm_page(output_capacity));
  if (output_addr == kAllocFailed) {
    csdvirt->csdvirt_release_memory(input_addr);
    return rocksdb::Status::Aborted("csdvirt_alloc_memory(output) failed");
  }
  auto t_alloc = clock::now();

  // csdvirt_load_files's `actual_sizes` is an INPUT, not an output, despite
  // the name -- confirmed by reading CSDVirt.cpp directly: it reads
  // actual_sizes[file] to know how many bytes to extract per file via
  // extent mapping (fiemap), and never writes a new value back anywhere in
  // the function. We previously zero-initialized it expecting the function
  // to fill in the real transferred size, which made leftover_size start at
  // 0 inside the function, immediately breaking its extent loop and
  // extracting nothing -- silently "succeeding" at loading 0 bytes every
  // time. The function's only real success/failure signal is its return
  // value (0 = success via the ioctl, -1 = failure), not this array.
  std::string file_list[1] = {file_path_};
  size_t actual_sizes[1] = {file_size_};
  size_t load_rc = csdvirt->csdvirt_load_files(file_list, 1, input_addr,
                                               actual_sizes);
  if (load_rc != 0) {
    nvmevirt_log(
        "RunMvccFilter: csdvirt_load_files failed (rc=%zd) for %s\n",
        (ssize_t)load_rc, file_path_.c_str());
    csdvirt->csdvirt_release_memory(input_addr);
    csdvirt->csdvirt_release_memory(output_addr);
    return rocksdb::Status::Aborted("csdvirt_load_files failed");
  }
  auto t_load = clock::now();

  struct CSD_PARAMS params;
  memset(&params, 0, sizeof(params));
  params.rocksdb_mvcc_filter_params.sstable_size = file_size_;
  params.rocksdb_mvcc_filter_params.snapshot_seq = snapshot_seq;
  params.rocksdb_mvcc_filter_params.output_capacity = output_capacity;

  // The device tracks output readiness for min(allocation, this value) bytes
  // and leaves the rest untracked, where a host read never comes ready. Our
  // output allocation is 2x the file, so pass that, not file_size_. The
  // program takes the real file length from sstable_size.
  const size_t task_span = align_up_to_slm_page(output_capacity);

  size_t result_len = 0;
  int rc = csdvirt->csdvirt_execute(ROCKSDB_MVCC_FILTER_PROGRAM_INDEX,
                                     input_addr, output_addr, task_span,
                                     &params, sizeof(params), &result_len);
  if (rc < 0) {
    nvmevirt_log("RunMvccFilter: csdvirt_execute failed (rc=%d) for %s\n", rc,
                 file_path_.c_str());
    csdvirt->csdvirt_release_memory(input_addr);
    csdvirt->csdvirt_release_memory(output_addr);
    return rocksdb::Status::Aborted("csdvirt_execute failed");
  }
  auto t_execute = clock::now();

  // Sync completes execute when the program finishes and returns the output
  // length. Async completes at dispatch and returns -1, arriving as
  // 0xFFFFFFFF through the 32-bit result field; the length then has to be
  // found by reading until the output runs out. output_capacity is bounded by
  // 2 * rocksdb_nvmevirt_max_sst_bytes, so a real length never reaches the
  // sentinel.
  constexpr size_t kNoResultSentinel = 0xFFFFFFFFu;
  constexpr size_t kHeaderLen = sizeof(struct rocksdb_mvcc_filter_output);
  const bool length_known = result_len != 0 &&
                            result_len != kNoResultSentinel &&
                            result_len <= output_capacity;

  const size_t read_limit = length_known ? result_len : output_capacity;
  std::unique_ptr<char[]> host_buf(new char[read_limit]);

  // csdvirt_read_slm() asserts on sizes above MAX_IO_SPLIT_SIZE, so split.
  // A short chunk means the device clamped against the end of the output.
  auto read_range = [&](size_t off, size_t want, size_t *got_out) -> bool {
    size_t done = 0;
    while (done < want) {
      size_t chunk = want - done;
      if (chunk > MAX_IO_SPLIT_SIZE) chunk = MAX_IO_SPLIT_SIZE;
      size_t got = csdvirt->csdvirt_read_slm(host_buf.get() + off + done,
                                             output_addr + off + done, chunk);
      if (got > chunk) return false;  // (size_t)-1 on ioctl failure
      done += got;
      if (got < chunk) break;
    }
    *got_out = done;
    return true;
  };

  size_t stream_len = 0;
  bool read_ok = false;
  if (length_known) {
    size_t got = 0;
    read_ok = read_range(0, result_len, &got);
    result_len = got;
    stream_len = (got > kHeaderLen) ? got - kHeaderLen : 0;
  } else {
    // Payload first, header last. The program writes the header at offset 0
    // only after emitting every entry, so an early read of it returns zeros.
    // End of output is reported only once the program has returned, so by
    // then the header is written.
    read_ok = read_range(kHeaderLen, output_capacity - kHeaderLen, &stream_len);
    if (read_ok) {
      size_t got = 0;
      read_ok = read_range(0, kHeaderLen, &got) && got == kHeaderLen;
    }
    result_len = kHeaderLen + stream_len;
  }

  if (!read_ok) {
    nvmevirt_log("RunMvccFilter: csdvirt_read_slm failed for %s\n",
                 file_path_.c_str());
    csdvirt->csdvirt_release_memory(input_addr);
    csdvirt->csdvirt_release_memory(output_addr);
    return rocksdb::Status::Aborted("csdvirt_read_slm failed");
  }
  auto t_read = clock::now();

  csdvirt->csdvirt_release_memory(input_addr);
  csdvirt->csdvirt_release_memory(output_addr);
  auto t_release = clock::now();

  if (result_len < kHeaderLen) {
    return rocksdb::Status::Corruption(
        "NvmeVirt mvcc_filter output shorter than its own header");
  }

  struct rocksdb_mvcc_filter_output header;
  memcpy(&header, host_buf.get(), kHeaderLen);
  *keys_seen = header.keys_seen;
  *keys_filtered = header.keys_filtered;

  char *stream_buf = new char[stream_len];
  memcpy(stream_buf, host_buf.get() + kHeaderLen, stream_len);

  *out_buf = stream_buf;
  *out_len = stream_len;

  auto us = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a)
        .count();
  };
  nvmevirt_log(
      "RunMvccFilter timing file=%s file_size=%llu result_len=%zu "
      "keys_seen=%llu keys_filtered=%llu total_us=%lld "
      "alloc_us=%lld load_us=%lld execute_us=%lld read_us=%lld "
      "release_us=%lld\n",
      file_path_.c_str(), (unsigned long long)file_size_, result_len,
      (unsigned long long)*keys_seen, (unsigned long long)*keys_filtered,
      us(t_start, t_release), us(t_start, t_alloc),
      us(t_alloc, t_load), us(t_load, t_execute), us(t_execute, t_read),
      us(t_read, t_release));

  return rocksdb::Status::OK();
}

rocksdb::InternalIterator *NvmeVirtTableReader::NewIterator(
    const rocksdb::ReadOptions &read_options,
    const rocksdb::SliceTransform *prefix_extractor, rocksdb::Arena *arena,
    bool skip_filters, rocksdb::TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  auto fallback = [&]() {
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller,
                               compaction_readahead_size,
                               allow_unprepared_value);
  };

  // Require the session to have opted in via
  // `SET SESSION rocksdb_nvmevirt_olap_session = 1`, not just the global
  // switch: RocksDB tags concurrent OLTP range reads on this CF kUserIterator
  // too, so without it they would all reach RunMvccFilter.
  THD *thd = current_thd;
  if (caller != rocksdb::kUserIterator || read_options.snapshot == nullptr ||
      !myrocks::rocksdb_nvmevirt_enabled || thd == nullptr ||
      !myrocks::rdb_nvmevirt_olap_session(thd)) {
    // Deliberately not logged: OLTP and compaction take this path millions of
    // times per session.
    return fallback();
  }
  if (file_size_ == 0 ||
      file_size_ > myrocks::rocksdb_nvmevirt_max_sst_bytes) {
    nvmevirt_log("NewIterator: guard rejected (file_size_) for %s\n",
                 file_path_.c_str());
    return fallback();
  }

  uint64_t snapshot_seq =
      static_cast<uint64_t>(read_options.snapshot->GetSequenceNumber());

  char *stream_buf = nullptr;
  size_t stream_len = 0;
  uint64_t keys_seen = 0, keys_filtered = 0;
  rocksdb::Status s = RunMvccFilter(snapshot_seq, &stream_buf, &stream_len,
                                    &keys_seen, &keys_filtered);
  if (!s.ok()) {
    nvmevirt_log(
        "NewIterator: CSD offload failed for %s: %s -- falling back to the "
        "unfiltered reader\n",
        file_path_.c_str(), s.ToString().c_str());
    return fallback();
  }

  nvmevirt_log(
      "NewIterator: SUCCESS for %s, keys_seen=%llu keys_filtered=%llu\n",
      file_path_.c_str(), (unsigned long long)keys_seen,
      (unsigned long long)keys_filtered);
  return new NvmeVirtResultIterator(stream_buf, stream_len, keys_seen,
                                    keys_filtered);
}

rocksdb::Status NvmeVirtTableFactory::NewTableReader(
    const rocksdb::ReadOptions &ro,
    const rocksdb::TableReaderOptions &table_reader_options,
    std::unique_ptr<rocksdb::RandomAccessFileReader> &&file,
    uint64_t file_size, std::unique_ptr<rocksdb::TableReader> *table_reader,
    bool prefetch_index_and_filter_in_cache) const {
  std::string file_path = file->file_name();

  std::unique_ptr<rocksdb::TableReader> inner_reader;
  rocksdb::Status s = inner_->NewTableReader(
      ro, table_reader_options, std::move(file), file_size, &inner_reader,
      prefetch_index_and_filter_in_cache);
  if (!s.ok()) {
    return s;
  }

  table_reader->reset(new NvmeVirtTableReader(
      std::move(inner_reader), std::move(file_path), file_size));
  return rocksdb::Status::OK();
}

#else  // !HAVE_NVMEVIRT

// Without -DWITH_NVMEVIRT=... at cmake configure time, NvmeVirtTableReader is
// a transparent passthrough and NvmeVirtTableFactory is never even
// instantiated (see rdb_cf_options.cc) -- but these two methods still need a
// definition since they're declared in the header unconditionally.

rocksdb::InternalIterator *NvmeVirtTableReader::NewIterator(
    const rocksdb::ReadOptions &read_options,
    const rocksdb::SliceTransform *prefix_extractor, rocksdb::Arena *arena,
    bool skip_filters, rocksdb::TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  return inner_->NewIterator(read_options, prefix_extractor, arena,
                             skip_filters, caller, compaction_readahead_size,
                             allow_unprepared_value);
}

rocksdb::Status NvmeVirtTableFactory::NewTableReader(
    const rocksdb::ReadOptions &ro,
    const rocksdb::TableReaderOptions &table_reader_options,
    std::unique_ptr<rocksdb::RandomAccessFileReader> &&file,
    uint64_t file_size, std::unique_ptr<rocksdb::TableReader> *table_reader,
    bool prefetch_index_and_filter_in_cache) const {
  return inner_->NewTableReader(ro, table_reader_options, std::move(file),
                                file_size, table_reader,
                                prefetch_index_and_filter_in_cache);
}

#endif  // HAVE_NVMEVIRT
