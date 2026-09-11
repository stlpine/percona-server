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

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "./nvmevirt_iterator.h"
#include "rocksdb/snapshot.h"
#include "rocksdb/status.h"
#include "rocksdb/table_properties.h"
#include "rocksdb/utilities/transaction_db.h"
#include "./ha_rocksdb_proto.h"
#include "./rdb_cf_manager.h"
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

// Grouped, tail-first offload. Each slot is loaded with two range loads:
//
//     load 1: file[tail_start .. size)  -> slot + 0          (index + footer)
//     load 2: file[0 .. tail_start)     -> slot + tail_len   (data blocks)
//
// The device's readiness frontier advances contiguously, so a backwards load
// would stall until the gap filled. Loading the tail first keeps every write
// forward-only and still gets the index in before the data blocks.
//
// Members come from one level below L0, where key ranges are disjoint, so no
// user key appears in two of them and grouping cannot change what gets
// filtered, only how the bytes move.

uint64_t NvmeVirtTableReader::TailStartOffset() const {
  std::shared_ptr<const rocksdb::TableProperties> props =
      inner_->GetTableProperties();
  if (props == nullptr) return 0;
  const uint64_t tail = props->tail_start_offset;
  if (tail == 0 || tail >= file_size_) return 0;
  // Move the seam down to a 512 boundary: csdvirt_load_part_of_file aligns its
  // source offset down and would otherwise skew the slot. The extra bytes are
  // the end of the last data block, which the rotated tail never reads.
  return tail & ~static_cast<uint64_t>(511);
}

std::vector<NvmeVirtTableReader::GroupFile> NvmeVirtTableReader::DiscoverGroup(
    size_t want) const {
  std::vector<GroupFile> group;
  const uint64_t self_tail = TailStartOffset();

  // A file whose tail offset is unknown can still be offloaded, just not
  // streamed: tail_start == size makes the device wait for the whole slot.
  group.push_back({file_path_, file_size_, self_tail ? self_tail : file_size_});
  if (want <= 1) return group;

  rocksdb::TransactionDB *db = myrocks::rdb_get_rocksdb_db();
  if (db == nullptr) return group;

  std::vector<rocksdb::LiveFileMetaData> all;
  db->GetLiveFilesMetaData(&all);

  const rocksdb::LiveFileMetaData *self = nullptr;
  for (const auto &f : all) {
    if (f.db_path + f.name == file_path_) {
      self = &f;
      break;
    }
  }
  // L0 files overlap in key range, so "the next files at this level" is not a
  // meaningful ordering there, and L0 is also where the filter yields nothing.
  if (self == nullptr || self->level == 0) return group;

  std::shared_ptr<rocksdb::ColumnFamilyHandle> cfh =
      myrocks::rdb_get_cf_manager().get_cf(self->column_family_name);
  if (cfh == nullptr) return group;

  // tail_start_offset is in the table properties, not the live file metadata.
  // GetPropertiesOfAllTables reads them through the table cache, which
  // max_open_files = -1 keeps populated, so this opens nothing.
  rocksdb::TablePropertiesCollection props;
  if (!db->GetPropertiesOfAllTables(cfh.get(), &props).ok()) return group;

  std::vector<const rocksdb::LiveFileMetaData *> peers;
  for (const auto &f : all) {
    if (f.level == self->level &&
        f.column_family_name == self->column_family_name) {
      peers.push_back(&f);
    }
  }
  std::sort(peers.begin(), peers.end(),
            [](const rocksdb::LiveFileMetaData *a,
               const rocksdb::LiveFileMetaData *b) {
              return a->smallestkey < b->smallestkey;
            });

  size_t self_idx = peers.size();
  for (size_t i = 0; i < peers.size(); i++) {
    if (peers[i]->db_path + peers[i]->name == file_path_) {
      self_idx = i;
      break;
    }
  }
  if (self_idx == peers.size()) return group;

  for (size_t i = self_idx + 1; i < peers.size() && group.size() < want; i++) {
    const std::string path = peers[i]->db_path + peers[i]->name;
    if (peers[i]->size == 0 ||
        peers[i]->size > myrocks::rocksdb_nvmevirt_max_sst_bytes) {
      break;
    }
    auto it = props.find(path);
    uint64_t tail = (it != props.end()) ? it->second->tail_start_offset : 0;
    if (tail == 0 || tail >= peers[i]->size) {
      tail = peers[i]->size;  // not streamable, still groupable
    } else {
      tail &= ~static_cast<uint64_t>(511);
    }
    group.push_back({path, peers[i]->size, tail});
  }
  return group;
}

rocksdb::Status NvmeVirtTableReader::RunMvccFilterGroup(
    uint64_t snapshot_seq, const std::vector<GroupFile> &files,
    std::vector<GroupStream> *streams) {
  using clock = std::chrono::steady_clock;
  auto t_start = clock::now();

  EnsureCsdvirtDevice();
  if (g_nvmevirt_csdvirt == nullptr) {
    return rocksdb::Status::Aborted("NvmeVirt CSD device unavailable");
  }
  if (files.empty() || files.size() > MAX_MVCC_GROUP_FILES) {
    return rocksdb::Status::InvalidArgument("bad group size");
  }
  CSDVirt *csdvirt = g_nvmevirt_csdvirt;

  constexpr size_t kSlmPageSize = 16 * 1024;
  auto align_up = [](size_t size) {
    return (size + kSlmPageSize - 1) & ~(kSlmPageSize - 1);
  };
  constexpr size_t kAllocFailed = static_cast<size_t>(-1);

  // Page-aligned slots, so a rotation seam never shares a page with the next
  // file's tail.
  std::vector<size_t> slot_offset(files.size());
  size_t input_bytes = 0, total_file_bytes = 0;
  for (size_t i = 0; i < files.size(); i++) {
    slot_offset[i] = input_bytes;
    input_bytes += align_up(files[i].size);
    total_file_bytes += files[i].size;
  }

  const size_t output_capacity =
      sizeof(struct rocksdb_mvcc_group_output) + 2 * total_file_bytes;

  size_t input_addr = csdvirt->csdvirt_alloc_memory(input_bytes);
  if (input_addr == kAllocFailed) {
    return rocksdb::Status::Aborted("csdvirt_alloc_memory(input) failed");
  }
  size_t output_addr = csdvirt->csdvirt_alloc_memory(align_up(output_capacity));
  if (output_addr == kAllocFailed) {
    csdvirt->csdvirt_release_memory(input_addr);
    return rocksdb::Status::Aborted("csdvirt_alloc_memory(output) failed");
  }
  auto t_alloc = clock::now();

  auto release = [&]() {
    csdvirt->csdvirt_release_memory(input_addr);
    csdvirt->csdvirt_release_memory(output_addr);
  };

  // Tail first, then data, in slot order, so destination addresses only rise.
  for (size_t i = 0; i < files.size(); i++) {
    std::string path[1] = {files[i].path};
    const uint64_t tail_len = files[i].size - files[i].tail_start;
    const uint64_t data_len = files[i].tail_start;
    const size_t slot = input_addr + slot_offset[i];

    if (tail_len > 0) {
      size_t sizes[1] = {static_cast<size_t>(tail_len)};
      if (csdvirt->csdvirt_load_part_of_file(path, 1, slot, sizes,
                                             files[i].tail_start,
                                             static_cast<int>(tail_len)) != 0) {
        nvmevirt_log("RunMvccFilterGroup: tail load failed for %s\n",
                     files[i].path.c_str());
        release();
        return rocksdb::Status::Aborted("tail load failed");
      }
    }
    if (data_len > 0) {
      size_t sizes[1] = {static_cast<size_t>(data_len)};
      if (csdvirt->csdvirt_load_part_of_file(path, 1, slot + tail_len, sizes, 0,
                                             static_cast<int>(data_len)) != 0) {
        nvmevirt_log("RunMvccFilterGroup: data load failed for %s\n",
                     files[i].path.c_str());
        release();
        return rocksdb::Status::Aborted("data load failed");
      }
    }
  }
  auto t_load = clock::now();

  struct CSD_PARAMS params;
  memset(&params, 0, sizeof(params));
  params.rocksdb_mvcc_group_params.num_files =
      static_cast<int>(files.size());
  params.rocksdb_mvcc_group_params.snapshot_seq = snapshot_seq;
  params.rocksdb_mvcc_group_params.output_capacity = output_capacity;
  params.rocksdb_mvcc_group_params.wait_whole_file =
      myrocks::rocksdb_nvmevirt_group_wait_whole;
  for (size_t i = 0; i < files.size(); i++) {
    params.rocksdb_mvcc_group_params.file_start_offset[i] = slot_offset[i];
    params.rocksdb_mvcc_group_params.file_size[i] = files[i].size;
    params.rocksdb_mvcc_group_params.tail_size[i] =
        files[i].size - files[i].tail_start;
  }

  size_t result_len = 0;
  int rc = csdvirt->csdvirt_execute(ROCKSDB_MVCC_GROUP_FILTER_PROGRAM_INDEX,
                                    input_addr, output_addr,
                                    align_up(output_capacity), &params,
                                    sizeof(params), &result_len);
  if (rc < 0) {
    nvmevirt_log("RunMvccFilterGroup: csdvirt_execute failed (rc=%d)\n", rc);
    release();
    return rocksdb::Status::Aborted("csdvirt_execute failed");
  }
  auto t_execute = clock::now();

  // Sync returns the real length; async completes at dispatch and returns the
  // 32-bit -1 sentinel.
  constexpr size_t kNoResultSentinel = 0xFFFFFFFFu;
  constexpr size_t kHeaderLen = sizeof(struct rocksdb_mvcc_group_output);
  const bool length_known = result_len != 0 &&
                            result_len != kNoResultSentinel &&
                            result_len <= output_capacity;
  const size_t read_limit = length_known ? result_len : output_capacity;
  std::unique_ptr<char[]> host_buf(new char[read_limit]);

  auto read_range = [&](size_t off, size_t want, size_t *got_out) -> bool {
    size_t done = 0;
    while (done < want) {
      size_t chunk = want - done;
      if (chunk > MAX_IO_SPLIT_SIZE) chunk = MAX_IO_SPLIT_SIZE;
      size_t got = csdvirt->csdvirt_read_slm(host_buf.get() + off + done,
                                             output_addr + off + done, chunk);
      if (got > chunk) return false;
      done += got;
      if (got < chunk) break;
    }
    *got_out = done;
    return true;
  };

  bool read_ok = false;
  size_t payload = 0;
  if (length_known) {
    size_t got = 0;
    read_ok = read_range(0, result_len, &got);
    result_len = got;
  } else {
    // Payload first, header last: the records are filled as each file
    // finishes, so an early header read returns zeros.
    read_ok = read_range(kHeaderLen, output_capacity - kHeaderLen, &payload);
    if (read_ok) {
      size_t got = 0;
      read_ok = read_range(0, kHeaderLen, &got) && got == kHeaderLen;
    }
    result_len = kHeaderLen + payload;
  }
  if (!read_ok) {
    nvmevirt_log("RunMvccFilterGroup: csdvirt_read_slm failed\n");
    release();
    return rocksdb::Status::Aborted("csdvirt_read_slm failed");
  }
  auto t_read = clock::now();
  release();

  if (result_len < kHeaderLen) {
    return rocksdb::Status::Corruption("NvmeVirt group output truncated");
  }

  struct rocksdb_mvcc_group_output header;
  memcpy(&header, host_buf.get(), kHeaderLen);
  if (header.num_files != files.size()) {
    return rocksdb::Status::Corruption("NvmeVirt group record count mismatch");
  }

  streams->clear();
  streams->reserve(files.size());
  for (size_t i = 0; i < files.size(); i++) {
    const uint64_t off = header.files[i].stream_offset;
    const uint64_t len = header.files[i].stream_len;
    if (off < kHeaderLen || off + len > result_len) {
      for (auto &st : *streams) delete[] st.buf;
      streams->clear();
      return rocksdb::Status::Corruption("NvmeVirt group record out of range");
    }
    char *buf = new char[len];
    memcpy(buf, host_buf.get() + off, len);
    streams->push_back({buf, static_cast<size_t>(len),
                        header.files[i].keys_seen,
                        header.files[i].keys_filtered});
  }

  auto us = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };
  nvmevirt_log(
      "RunMvccFilterGroup timing files=%zu input_bytes=%zu result_len=%zu "
      "stream=%d total_us=%lld alloc_us=%lld load_us=%lld execute_us=%lld "
      "read_us=%lld\n",
      files.size(), total_file_bytes, result_len,
      myrocks::rocksdb_nvmevirt_group_wait_whole ? 0 : 1,
      us(t_start, t_read), us(t_start, t_alloc), us(t_alloc, t_load),
      us(t_load, t_execute), us(t_execute, t_read));

  return rocksdb::Status::OK();
}

namespace {

// Streams for group members not asked for yet. thread_local because one scan
// runs on one connection: the group is filtered once and each file's iterator
// collects its own slice. Without this a group of 4 does four times the work
// and discards three quarters of it.
struct PendingStream {
  uint64_t snapshot_seq;
  NvmeVirtTableReader::GroupStream stream;
};

thread_local std::map<std::string, PendingStream> g_pending_streams;

}  // namespace

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

  // 0 keeps the original whole-file command, the arm every archived
  // measurement used. 1 streams one file per command; 2 to 4 also overlap the
  // next file's load with this file's compute.
  const unsigned long group_size = myrocks::rocksdb_nvmevirt_group_size;
  if (group_size >= 1) {
    auto claim = [&](GroupStream *out) -> bool {
      auto it = g_pending_streams.find(file_path_);
      if (it == g_pending_streams.end()) return false;
      const bool usable = it->second.snapshot_seq == snapshot_seq;
      if (usable) {
        *out = it->second.stream;
      } else {
        // A different snapshot needs different filtering; the stream is dead.
        delete[] it->second.stream.buf;
      }
      g_pending_streams.erase(it);
      return usable;
    };

    GroupStream mine;
    bool have_mine = claim(&mine);

    if (!have_mine) {
      std::vector<GroupFile> group = DiscoverGroup(
          group_size > MAX_MVCC_GROUP_FILES ? MAX_MVCC_GROUP_FILES
                                            : group_size);
      std::vector<GroupStream> streams;
      rocksdb::Status gs = RunMvccFilterGroup(snapshot_seq, group, &streams);
      if (!gs.ok() || streams.size() != group.size()) {
        for (auto &st : streams) delete[] st.buf;
        nvmevirt_log(
            "NewIterator: grouped offload failed for %s: %s, falling back\n",
            file_path_.c_str(), gs.ToString().c_str());
        return fallback();
      }
      mine = streams[0];
      have_mine = true;
      // Hold the siblings for the iterators RocksDB will open on them.
      for (size_t i = 1; i < streams.size(); i++) {
        auto prev = g_pending_streams.find(group[i].path);
        if (prev != g_pending_streams.end()) {
          delete[] prev->second.stream.buf;
          g_pending_streams.erase(prev);
        }
        g_pending_streams[group[i].path] = {snapshot_seq, streams[i]};
      }
    }

    nvmevirt_log(
        "NewIterator: SUCCESS (group) for %s, keys_seen=%llu keys_filtered=%llu\n",
        file_path_.c_str(), (unsigned long long)mine.keys_seen,
        (unsigned long long)mine.keys_filtered);
    return new NvmeVirtResultIterator(mine.buf, mine.len, mine.keys_seen,
                                      mine.keys_filtered);
  }

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
