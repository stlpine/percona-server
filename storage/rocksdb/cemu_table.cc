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

#include "./cemu_table.h"

#include "./cemu_iterator.h"

#include "rocksdb/table_reader_caller.h"

// CEMU ioctl headers are only present inside the CEMU VM build environment.
// When HAVE_CEMU is not defined the class compiles but NewIterator() always
// falls back to the unfiltered inner iterator, making the wrapper a transparent
// no-op.  Set -DHAVE_CEMU in the cmake configure step inside the CEMU VM.
#ifdef HAVE_CEMU
#include <fcntl.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>  // aligned_alloc, free
#include "cemu_ioctl.h"  // ioctl_download, ioctl_execute, ioctl_create_mrs,
                         // IOCTL_CEMU_{DOWNLOAD,ACTIVATE,EXECUTE,
                         //             DEACTIVATE,CREATE_MRS,DELETE_MRS}
// PROGRAM_TYPE_SHARED_LIB is defined in the CEMU test util.h enum; mirror it
// here so we don't depend on that header.
#ifndef PROGRAM_TYPE_SHARED_LIB
#define PROGRAM_TYPE_SHARED_LIB 2
#endif
#endif  // HAVE_CEMU

#include <atomic>
#include <cstring>
#include <mutex>

// Path inside the CEMU VM where mvcc_filter.so is deployed.
#ifndef CEMU_CSF_SO_PATH
#define CEMU_CSF_SO_PATH "/opt/cemu_csf/mvcc_filter.so"
#endif

// Compute namespace device node exposed by CEMU.
#ifndef CEMU_COMPUTE_DEV
#define CEMU_COMPUTE_DEV "/dev/nvme0c3"
#endif

// FDM mount point — on-device DRAM filesystem.
#ifndef CEMU_FDM_MOUNT
#define CEMU_FDM_MOUNT "/mnt/fdm0"
#endif

// Output buffer header written by mvcc_filter CSF (16 bytes total):
//   bytes 0-7:  result_bytes — number of KV-stream bytes that follow
//   bytes 8-15: keys_seen   — total internal keys examined by the CSF
// The KV stream starts at offset 16.
static const size_t kCsfHeaderSize = 16;

namespace myrocks {

// Definitions live in ha_rocksdb.cc; declared extern in cemu_iterator.h.
extern std::atomic<uint64_t> rocksdb_cemu_keys_seen;
extern std::atomic<uint64_t> rocksdb_cemu_keys_filtered;
extern bool rocksdb_cemu_enabled;

}  // namespace myrocks

// ---------------------------------------------------------------------------
// CemuTableFactory::NewTableReader
// ---------------------------------------------------------------------------

rocksdb::Status CemuTableFactory::NewTableReader(
    const rocksdb::ReadOptions &ro,
    const rocksdb::TableReaderOptions &table_reader_options,
    std::unique_ptr<rocksdb::RandomAccessFileReader> &&file, uint64_t file_size,
    std::unique_ptr<rocksdb::TableReader> *table_reader,
    bool prefetch_index_and_filter_in_cache) const {
  // Capture file metadata before moving the reader into inner_.
  std::string file_path = file->file_name();
  uint64_t file_number = table_reader_options.cur_file_num;

  std::unique_ptr<rocksdb::TableReader> inner_reader;
  rocksdb::Status s = inner_->NewTableReader(
      ro, table_reader_options, std::move(file), file_size, &inner_reader,
      prefetch_index_and_filter_in_cache);
  if (!s.ok()) return s;

  *table_reader = std::make_unique<CemuTableReader>(
      std::move(inner_reader), std::move(file_path), file_number);
  return rocksdb::Status::OK();
}

// ---------------------------------------------------------------------------
// CemuTableReader
// ---------------------------------------------------------------------------

CemuTableReader::~CemuTableReader() {
#ifdef HAVE_CEMU
  if (csf_ready_ && cemu_fd_ >= 0) {
    struct ioctl_download pi{};
    pi.pind = pind_;
    ioctl(cemu_fd_, IOCTL_CEMU_DEACTIVATE, &pi);
    close(cemu_fd_);
  }
#endif
}

bool CemuTableReader::EnsureCsfLoaded() {
#ifndef HAVE_CEMU
  return false;
#else
  std::call_once(csf_once_, [this]() {
    cemu_fd_ = open(CEMU_COMPUTE_DEV, O_RDWR);
    if (cemu_fd_ < 0) return;

    // Pack "<so_path>\0<func_name>\0" into a 4096-aligned buffer.
    // The CSD kernel driver reads the .so by path from the shared filesystem
    // (FDMFS or 9p mount), so we pass the path string — not the binary.
    const char *func_name = "mvcc_filter";
    const size_t path_len = strlen(CEMU_CSF_SO_PATH);
    const size_t func_len = strlen(func_name);
    const size_t buf_size = path_len + 1 + func_len + 1;
    const size_t alloc_size = (buf_size + 4095) & ~static_cast<size_t>(4095);
    char *prog_buf = static_cast<char *>(aligned_alloc(4096, alloc_size));
    if (!prog_buf) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }
    memcpy(prog_buf, CEMU_CSF_SO_PATH, path_len + 1);
    memcpy(prog_buf + path_len + 1, func_name, func_len + 1);

    struct ioctl_download dl{};
    dl.addr = prog_buf;
    dl.size = static_cast<int32_t>(buf_size);
    dl.ptype = PROGRAM_TYPE_SHARED_LIB;
    const int dl_ret = ioctl(cemu_fd_, IOCTL_CEMU_DOWNLOAD, &dl);
    free(prog_buf);
    if (dl_ret < 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    // Activate the downloaded program; pind from download is the program index.
    struct ioctl_download act{};
    act.pind = dl.pind;
    if (ioctl(cemu_fd_, IOCTL_CEMU_ACTIVATE, &act) < 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    pind_ = dl.pind;
    csf_ready_ = true;
  });
  return csf_ready_;
#endif  // HAVE_CEMU
}

rocksdb::InternalIterator *CemuTableReader::NewIterator(
    const rocksdb::ReadOptions &read_options,
    const rocksdb::SliceTransform *prefix_extractor, rocksdb::Arena *arena,
    bool skip_filters, rocksdb::TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  // Guard: only wrap user-facing forward scans with an explicit snapshot.
  // Compaction, flush, and point-lookup paths must see all versions.
  if (caller != rocksdb::kUserIterator || read_options.snapshot == nullptr ||
      !myrocks::rocksdb_cemu_enabled) {
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

#ifndef HAVE_CEMU
  // CEMU not compiled in — transparent fallback.
  return inner_->NewIterator(read_options, prefix_extractor, arena,
                             skip_filters, caller, compaction_readahead_size,
                             allow_unprepared_value);
#else
  if (!EnsureCsfLoaded()) {
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  const rocksdb::SequenceNumber snap_seq =
      read_options.snapshot->GetSequenceNumber();

  // Open the SST file so the CSD can map it as the MRS input region.
  int sst_fd = open(file_path_.c_str(), O_RDWR);
  if (sst_fd < 0) {
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }
  struct stat sst_st{};
  if (fstat(sst_fd, &sst_st) != 0 || sst_st.st_size == 0) {
    close(sst_fd);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }
  const uint64_t file_size = static_cast<uint64_t>(sst_st.st_size);

  // FDM output buffer path — unique per file + thread to support concurrent
  // scans of the same SST from multiple OLAP threads.
  char out_path[256];
  snprintf(out_path, sizeof(out_path), CEMU_FDM_MOUNT "/%llu_%lu.out",
           (unsigned long long)file_number_, (unsigned long)pthread_self());

  // Output capacity: header (16 bytes) + up to file_size bytes of KV stream.
  const uint64_t out_capacity = kCsfHeaderSize + file_size;

  int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (out_fd < 0) {
    close(sst_fd);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }
  // Pre-allocate the output file so FDMFS can back it with device DRAM.
  if (ftruncate(out_fd, static_cast<off_t>(out_capacity)) != 0) {
    close(out_fd);
    close(sst_fd);
    unlink(out_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Create Memory Range Set: mr[0] = SST input, mr[1] = FDM output.
  // ioctl_create_mrs uses pointer arrays for fds / offsets / sizes.
  int     mrs_fds[2]   = { sst_fd, out_fd };
  long long mrs_offs[2] = { 0, 0 };
  long long mrs_sizes[2] = { static_cast<long long>(file_size),
                              static_cast<long long>(out_capacity) };

  struct ioctl_create_mrs mrs{};
  mrs.nr_fd = 2;
  mrs.fd   = mrs_fds;
  mrs.off  = mrs_offs;
  mrs.size = mrs_sizes;
  if (ioctl(cemu_fd_, IOCTL_CEMU_CREATE_MRS, &mrs) < 0) {
    close(out_fd);
    close(sst_fd);
    unlink(out_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Execute the CSF synchronously.
  struct ioctl_execute exec{};
  exec.pind   = static_cast<uint16_t>(pind_);
  exec.rsid   = mrs.rsid;
  exec.cparam1 = static_cast<uint64_t>(snap_seq);
  if (ioctl(cemu_fd_, IOCTL_CEMU_EXECUTE, &exec) < 0) {
    struct ioctl_create_mrs del{};
    del.rsid = mrs.rsid;
    ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &del);
    close(out_fd);
    close(sst_fd);
    unlink(out_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // MRS can be released immediately after execution completes.
  {
    struct ioctl_create_mrs del{};
    del.rsid = mrs.rsid;
    ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &del);
  }
  close(sst_fd);

  // Read the 16-byte header written by mvcc_filter:
  //   bytes 0-7:  result_bytes (KV stream length)
  //   bytes 8-15: keys_seen
  char hdr[kCsfHeaderSize] = {};
  uint64_t result_bytes = 0;
  uint64_t keys_seen    = 0;
  if (pread(out_fd, hdr, kCsfHeaderSize, 0) ==
      static_cast<ssize_t>(kCsfHeaderSize)) {
    memcpy(&result_bytes, hdr,     8);
    memcpy(&keys_seen,    hdr + 8, 8);
  }

  // Read the KV stream that follows the header.
  char *result_buf = nullptr;
  if (result_bytes > 0) {
    result_buf = new char[result_bytes];
    if (pread(out_fd, result_buf, result_bytes,
              static_cast<off_t>(kCsfHeaderSize)) !=
        static_cast<ssize_t>(result_bytes)) {
      delete[] result_buf;
      result_buf = nullptr;
    }
  }
  close(out_fd);
  unlink(out_path);

  if (!result_buf && result_bytes > 0) {
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Count emitted entries by walking the flat result stream.
  uint64_t keys_emitted = 0;
  {
    size_t scan = 0;
    while (scan + 8 <= result_bytes) {
      uint32_t klen, vlen;
      memcpy(&klen, result_buf + scan, 4);
      scan += 4;
      if (scan + klen + 4 > result_bytes) break;
      scan += klen;
      memcpy(&vlen, result_buf + scan, 4);
      scan += 4;
      if (scan + vlen > result_bytes) break;
      scan += vlen;
      ++keys_emitted;
    }
  }
  const uint64_t keys_filtered =
      keys_seen > keys_emitted ? keys_seen - keys_emitted : 0;

  return new CemuResultIterator(result_buf, result_bytes, keys_seen,
                                keys_filtered);
#endif  // HAVE_CEMU
}
