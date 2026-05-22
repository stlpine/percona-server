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
#include <vector>
#include "cemu_nvme.h"  // CEMU-provided: cemu_prog_info, cemu_mrs_info,
                        //               IOCTL_CEMU_{DOWNLOAD,ACTIVATE,
                        //                           EXECUTE,DEACTIVATE,
                        //                           CREATE_MRS,DELETE_MRS}
#endif                  // HAVE_CEMU

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
    struct cemu_prog_info pi{};
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

    // Read the .so into memory and send to the CSD.
    int so_fd = open(CEMU_CSF_SO_PATH, O_RDONLY);
    if (so_fd < 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    struct stat st{};
    if (fstat(so_fd, &st) != 0 || st.st_size == 0) {
      close(so_fd);
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    std::vector<char> so_buf(st.st_size);
    if (read(so_fd, so_buf.data(), st.st_size) != st.st_size) {
      close(so_fd);
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }
    close(so_fd);

    struct cemu_prog_info dl{};
    dl.prog_type = CEMU_PROG_SHARED_LIB;
    dl.prog_data = so_buf.data();
    dl.prog_size = static_cast<uint32_t>(st.st_size);
    strncpy(dl.func_name, "mvcc_filter", sizeof(dl.func_name) - 1);
    if (ioctl(cemu_fd_, IOCTL_CEMU_DOWNLOAD, &dl) != 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    struct cemu_prog_info act{};
    act.prog_id = dl.prog_id;
    if (ioctl(cemu_fd_, IOCTL_CEMU_ACTIVATE, &act) != 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    pind_ = act.pind;
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

  // Open the full SST file so the CSD can map it into the MRS.
  // The CSF parses the footer + index block to locate individual data blocks.
  int sst_fd = open(file_path_.c_str(), O_RDONLY);
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

  int out_fd = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (out_fd < 0) {
    close(sst_fd);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Create Memory Range Set mapping [SST data blocks] → [FDM output buffer].
  struct cemu_mrs_info mrs{};
  mrs.mr[0].fd = sst_fd;
  mrs.mr[0].offset = 0;
  mrs.mr[0].size = file_size;
  mrs.mr[1].fd = out_fd;
  mrs.mr[1].offset = 0;
  mrs.mr[1].size = file_size;  // filtered output ≤ input size
  mrs.nr_mr = 2;
  if (ioctl(cemu_fd_, IOCTL_CEMU_CREATE_MRS, &mrs) != 0) {
    close(out_fd);
    close(sst_fd);
    unlink(out_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Execute the CSF synchronously.
  struct cemu_exec_info exec{};
  exec.pind = pind_;
  exec.rsid = mrs.rsid;
  exec.cparam1 = static_cast<long long>(snap_seq);
  if (ioctl(cemu_fd_, IOCTL_CEMU_EXECUTE, &exec) != 0) {
    ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &mrs);
    close(out_fd);
    close(sst_fd);
    unlink(out_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // MRS can be released immediately after execution completes.
  ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &mrs);
  close(sst_fd);

  // Read the flat KV stream from the FDM output buffer.
  const size_t result_bytes = static_cast<size_t>(exec.ret);
  char *result_buf = nullptr;
  if (result_bytes > 0) {
    result_buf = new char[result_bytes];
    if (pread(out_fd, result_buf, result_bytes, 0) !=
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

  // The CSF writes keys_seen to exec.cparam2 (args->cparam2 is writable by
  // CSF). Count emitted entries by walking the flat result stream.
  uint64_t keys_seen = static_cast<uint64_t>(exec.cparam2);
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
  uint64_t keys_filtered =
      keys_seen > keys_emitted ? keys_seen - keys_emitted : 0;

  return new CemuResultIterator(result_buf, result_bytes, keys_seen,
                                keys_filtered);
#endif  // HAVE_CEMU
}
