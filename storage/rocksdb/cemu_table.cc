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
#include <linux/nvme_ioctl.h>  // nvme_passthru_cmd, NVME_IOCTL_IO_CMD
#include "cemu_ioctl.h"  // ioctl_download, ioctl_create_mrs,
                         // IOCTL_CEMU_{DOWNLOAD,ACTIVATE,DEACTIVATE,
                         //             CREATE_MRS,DELETE_MRS}
#ifndef PROGRAM_TYPE_SHARED_LIB
#define PROGRAM_TYPE_SHARED_LIB 2
#endif

// NVMe vendor command layout for CEMU program execution.
// Overlays the first 64 bytes of struct nvme_passthru_cmd.
// Matches struct nvme_program_execute_cmd in CEMU/tests/cemu/util.h.
struct nvme_program_execute_cmd {
  uint8_t  opcode;
  uint8_t  flags;
  uint16_t cid;
  uint32_t nsid;
  uint16_t pind;
  uint16_t rsid;
  uint32_t numr;
  uint32_t dlen;
  uint32_t rsvd;
  uint64_t prp1;
  uint64_t prp2;
  uint64_t cparam1;
  uint64_t cparam2;
  uint32_t group     : 8;
  uint32_t chunk_nlb : 24;
  uint32_t user_runtime;
};
static_assert(sizeof(nvme_program_execute_cmd) == 64,
              "nvme_program_execute_cmd must be 64 bytes");
#endif  // HAVE_CEMU

#include <atomic>
#include <cstring>
#include <mutex>
#include <stdio.h>

// Temporary debug logger — writes to /tmp/cemu_debug.log.
// Remove before production use.
static void cemu_log(const char *fmt, ...) {
  FILE *f = fopen("/tmp/cemu_debug.log", "a");
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fputc('\n', f);
  fclose(f);
}
#include <stdarg.h>

// Global mutex serialising all CEMU executions.
// FDMFS pre-allocates a fixed set of files; concurrent writers would corrupt
// each other's data.  A single pair (input=0, output=1) is used for now.
static std::mutex g_cemu_exec_mutex;

// Path inside the CEMU VM where mvcc_filter.so is deployed.
#ifndef CEMU_CSF_SO_PATH
#define CEMU_CSF_SO_PATH "/opt/cemu_csf/mvcc_filter.so"
#endif

// Compute namespace device node exposed by CEMU.
#ifndef CEMU_COMPUTE_DEV
#define CEMU_COMPUTE_DEV "/dev/nvme0c3"
#endif

// NVMe generic passthru device for program execution (namespace 3).
#ifndef CEMU_NG_DEV
#define CEMU_NG_DEV "/dev/ng0n3"
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
  cemu_log("NewTableReader: wrapped file_number=%llu", (unsigned long long)file_number);
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
  if (ng_fd_ >= 0) close(ng_fd_);
#endif
}

bool CemuTableReader::EnsureCsfLoaded() {
#ifndef HAVE_CEMU
  return false;
#else
  std::call_once(csf_once_, [this]() {
    cemu_fd_ = open(CEMU_COMPUTE_DEV, O_RDWR);
    cemu_log("EnsureCsfLoaded: open(%s) fd=%d errno=%d", CEMU_COMPUTE_DEV, cemu_fd_, errno);
    if (cemu_fd_ < 0) return;

    const char *func_name = "mvcc_filter";

    // FEMU's load_shared_lib interprets dl.addr as a HOST filesystem path string
    // to dlopen() — it does not accept raw ELF bytes.  Pass the path + function
    // name as two consecutive null-terminated strings in the buffer.
    const size_t path_len = strlen(CEMU_CSF_SO_PATH);
    const size_t func_len = strlen(func_name);
    const size_t buf_size = path_len + 1 + func_len + 1;
    const size_t alloc_size = (buf_size + 4095) & ~static_cast<size_t>(4095);
    char *prog_buf = static_cast<char *>(aligned_alloc(4096, alloc_size));
    if (!prog_buf) {
      cemu_log("EnsureCsfLoaded: aligned_alloc failed");
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }
    memset(prog_buf, 0, alloc_size);
    memcpy(prog_buf, CEMU_CSF_SO_PATH, path_len + 1);
    memcpy(prog_buf + path_len + 1, func_name, func_len + 1);
    cemu_log("EnsureCsfLoaded: downloading so_path=%s func=%s buf_size=%zu",
             CEMU_CSF_SO_PATH, func_name, buf_size);

    struct ioctl_download dl{};
    dl.name = func_name;  // required — DOWNLOAD returns EFAULT if name is null
    dl.addr = prog_buf;
    dl.size = static_cast<int32_t>(buf_size);
    dl.ptype = PROGRAM_TYPE_SHARED_LIB;
    int dl_ret = ioctl(cemu_fd_, IOCTL_CEMU_DOWNLOAD, &dl);
    cemu_log("EnsureCsfLoaded: DOWNLOAD ret=%d pind=%d errno=%d", dl_ret, (int)dl.pind, errno);

    if (dl_ret < 0 && errno == EEXIST) {
      // Stale slot from a previous failed load — unload it and retry once.
      cemu_log("EnsureCsfLoaded: EEXIST pind=%d, unloading stale slot", (int)dl.pind);
      struct ioctl_download unload{};
      unload.name = func_name;
      unload.pind = dl.pind;
      ioctl(cemu_fd_, IOCTL_CEMU_UNLOAD, &unload);

      memset(&dl, 0, sizeof(dl));
      dl.name = func_name;
      dl.addr = prog_buf;
      dl.size = static_cast<int32_t>(buf_size);
      dl.ptype = PROGRAM_TYPE_SHARED_LIB;
      dl_ret = ioctl(cemu_fd_, IOCTL_CEMU_DOWNLOAD, &dl);
      cemu_log("EnsureCsfLoaded: DOWNLOAD retry ret=%d pind=%d errno=%d",
               dl_ret, (int)dl.pind, errno);
    }

    free(prog_buf);
    if (dl_ret < 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    struct ioctl_download act{};
    act.pind = dl.pind;
    const int act_ret = ioctl(cemu_fd_, IOCTL_CEMU_ACTIVATE, &act);
    cemu_log("EnsureCsfLoaded: ACTIVATE ret=%d errno=%d", act_ret, errno);
    if (act_ret < 0) {
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    pind_ = dl.pind;

    ng_fd_ = open(CEMU_NG_DEV, O_RDWR);
    cemu_log("EnsureCsfLoaded: open(%s) ng_fd=%d errno=%d", CEMU_NG_DEV, ng_fd_, errno);
    if (ng_fd_ < 0) {
      struct ioctl_download deact{};
      deact.pind = pind_;
      ioctl(cemu_fd_, IOCTL_CEMU_DEACTIVATE, &deact);
      close(cemu_fd_);
      cemu_fd_ = -1;
      return;
    }

    csf_ready_ = true;
    cemu_log("EnsureCsfLoaded: ready pind=%d ng_fd=%d", pind_, ng_fd_);
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
  cemu_log("NewIterator: caller=%d snapshot=%p cemu_enabled=%d file=%s",
           (int)caller, (void*)read_options.snapshot,
           (int)myrocks::rocksdb_cemu_enabled, file_path_.c_str());
  if (caller != rocksdb::kUserIterator || read_options.snapshot == nullptr ||
      !myrocks::rocksdb_cemu_enabled) {
    cemu_log("NewIterator: guard fallback caller=%d snap_null=%d enabled=%d",
             (int)caller, read_options.snapshot == nullptr ? 1 : 0,
             (int)myrocks::rocksdb_cemu_enabled);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

#ifndef HAVE_CEMU
  // CEMU not compiled in — transparent fallback.
  cemu_log("NewIterator: HAVE_CEMU not defined, fallback");
  return inner_->NewIterator(read_options, prefix_extractor, arena,
                             skip_filters, caller, compaction_readahead_size,
                             allow_unprepared_value);
#else
  if (!EnsureCsfLoaded()) {
    cemu_log("NewIterator: EnsureCsfLoaded failed, fallback");
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }
  cemu_log("NewIterator: CEMU path active pind=%d", pind_);

  const rocksdb::SequenceNumber snap_seq =
      read_options.snapshot->GetSequenceNumber();

  // Serialize all CEMU executions — FDMFS pre-allocates a fixed file pool;
  // concurrent writers would overlap.
  std::lock_guard<std::mutex> cemu_lock(g_cemu_exec_mutex);

  // Open SST file and measure its size.
  int sst_fd = open(file_path_.c_str(), O_RDONLY);
  if (sst_fd < 0) {
    cemu_log("NewIterator: open SST failed errno=%d", errno);
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
  const uint64_t out_capacity = kCsfHeaderSize + file_size;

  // Use pre-existing FDMFS files — FDMFS does not support writes to
  // dynamically created files (kernel NULL deref in fdmfs_iomap_begin).
  // Files 0 and 1 are pre-allocated at mount time (32 MB each).
  const char *fdm_in_path  = CEMU_FDM_MOUNT "/0";
  const char *fdm_out_path = CEMU_FDM_MOUNT "/1";

  int fdm_in_fd = open(fdm_in_path, O_RDWR);
  if (fdm_in_fd < 0) {
    cemu_log("NewIterator: open FDM input failed errno=%d", errno);
    close(sst_fd);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // Copy SST content into FDM input file at offset 0.
  // FDMFS uses iomap_dio_rw for all I/O: buffer and size must be 4096-aligned.
  {
    const size_t aligned_size = (file_size + 4095) & ~static_cast<size_t>(4095);
    char *copy_buf = static_cast<char *>(aligned_alloc(4096, aligned_size));
    bool copy_ok = false;
    if (copy_buf) {
      memset(copy_buf, 0, aligned_size);
      ssize_t nr = pread(sst_fd, copy_buf, file_size, 0);
      if (nr == static_cast<ssize_t>(file_size)) {
        ssize_t nw = pwrite(fdm_in_fd, copy_buf, aligned_size, 0);
        copy_ok = (nw == static_cast<ssize_t>(aligned_size));
        if (!copy_ok)
          cemu_log("NewIterator: FDM pwrite failed nw=%zd errno=%d", nw, errno);
      } else {
        cemu_log("NewIterator: SST pread failed nr=%zd errno=%d", nr, errno);
      }
      free(copy_buf);
    }
    if (!copy_ok) {
      close(fdm_in_fd); close(sst_fd);
      return inner_->NewIterator(read_options, prefix_extractor, arena,
                                 skip_filters, caller, compaction_readahead_size,
                                 allow_unprepared_value);
    }
  }
  close(sst_fd);
  cemu_log("NewIterator: SST staged to FDM input file size=%llu",
           (unsigned long long)file_size);

  int fdm_out_fd = open(fdm_out_path, O_RDWR);
  if (fdm_out_fd < 0) {
    cemu_log("NewIterator: open FDM output failed errno=%d", errno);
    close(fdm_in_fd);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }

  // CREATE_MRS: mr[0]=FDM input (SST copy), mr[1]=FDM output.
  int       mrs_fds[2]   = { fdm_in_fd, fdm_out_fd };
  long long mrs_offs[2]  = { 0, 0 };
  long long mrs_sizes[2] = { static_cast<long long>(file_size),
                              static_cast<long long>(out_capacity) };
  struct ioctl_create_mrs mrs{};
  mrs.nr_fd = 2;
  mrs.fd    = mrs_fds;
  mrs.off   = mrs_offs;
  mrs.size  = mrs_sizes;
  if (ioctl(cemu_fd_, IOCTL_CEMU_CREATE_MRS, &mrs) < 0) {
    cemu_log("NewIterator: CREATE_MRS failed errno=%d", errno);
    close(fdm_out_fd); unlink(fdm_out_path);
    close(fdm_in_fd); unlink(fdm_in_path);
    return inner_->NewIterator(read_options, prefix_extractor, arena,
                               skip_filters, caller, compaction_readahead_size,
                               allow_unprepared_value);
  }
  cemu_log("NewIterator: CREATE_MRS ok rsid=%d", (int)mrs.rsid);

  // Execute CSF via NVMe passthru on /dev/ng0n3 (same mechanism as vadd_example).
  // nvme_program_execute_cmd overlays the first 64 bytes of nvme_passthru_cmd.
  {
    struct nvme_passthru_cmd nvme_cmd{};
    struct nvme_program_execute_cmd *exec =
        reinterpret_cast<struct nvme_program_execute_cmd *>(&nvme_cmd);
    exec->opcode  = 0x01;
    exec->nsid    = 3;
    exec->pind    = static_cast<uint16_t>(pind_);
    exec->rsid    = mrs.rsid;
    exec->cparam1 = static_cast<uint64_t>(snap_seq);
    cemu_log("NewIterator: EXECUTE pind=%d rsid=%d snap_seq=%llu",
             (int)pind_, (int)mrs.rsid, (unsigned long long)snap_seq);
    const int exec_ret = ioctl(ng_fd_, NVME_IOCTL_IO_CMD, &nvme_cmd);
    cemu_log("NewIterator: EXECUTE ret=%d errno=%d", exec_ret, errno);
    if (exec_ret < 0) {
      struct ioctl_create_mrs del{}; del.rsid = mrs.rsid;
      ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &del);
      close(fdm_out_fd);
      close(fdm_in_fd);
      return inner_->NewIterator(read_options, prefix_extractor, arena,
                                 skip_filters, caller, compaction_readahead_size,
                                 allow_unprepared_value);
    }
  }

  // Release MRS — FDM files are permanent pool members, not deleted.
  {
    struct ioctl_create_mrs del{}; del.rsid = mrs.rsid;
    ioctl(cemu_fd_, IOCTL_CEMU_DELETE_MRS, &del);
  }
  close(fdm_in_fd);

  // Read the 16-byte header written by mvcc_filter into the FDM output file:
  //   bytes 0-7:  result_bytes (KV stream length)
  //   bytes 8-15: keys_seen
  char hdr[kCsfHeaderSize] = {};
  uint64_t result_bytes = 0;
  uint64_t keys_seen    = 0;
  if (pread(fdm_out_fd, hdr, kCsfHeaderSize, 0) ==
      static_cast<ssize_t>(kCsfHeaderSize)) {
    memcpy(&result_bytes, hdr,     8);
    memcpy(&keys_seen,    hdr + 8, 8);
  }
  cemu_log("NewIterator: result_bytes=%llu keys_seen=%llu",
           (unsigned long long)result_bytes, (unsigned long long)keys_seen);

  // Read the KV stream that follows the header.
  char *result_buf = nullptr;
  if (result_bytes > 0) {
    result_buf = new char[result_bytes];
    if (pread(fdm_out_fd, result_buf, result_bytes,
              static_cast<off_t>(kCsfHeaderSize)) !=
        static_cast<ssize_t>(result_bytes)) {
      delete[] result_buf;
      result_buf = nullptr;
    }
  }
  close(fdm_out_fd);

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
