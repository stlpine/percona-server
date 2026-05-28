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

// mvcc_filter.cc — CEMU Computational Storage Function (CSF)
//
// Compiled as a standalone shared library (.so) and executed on the emulated
// CSD compute unit inside CEMU. Never linked into mysqld.
//
// Memory Range Set layout (set by CemuTableReader before CSF execution):
//   mr[0]: input  — full SST file bytes (mapped from NVM namespace)
//   mr[1]: output — FDM buffer: 16-byte header followed by flat KV stream
//
// cparam1 (in): snapshot_seq (uint64_t)
//
// Output buffer layout (mr[1]):
//   bytes  0-7:  result_bytes — number of KV-stream bytes that follow
//   bytes 8-15:  keys_seen   — total internal keys examined
//   bytes 16+:   [uint32 key_len][key bytes][uint32 val_len][value bytes] ...
//
// IOCTL_CEMU_EXECUTE is _IOW (write-only): the ioctl struct is NOT copied back
// to userspace after execution.  All output must be encoded in mr[1].
//
// Compile (inside CEMU VM):
//   g++ -shared -fPIC -O2 -std=c++17 \
//       -DHAVE_CEMU_SDK \
//       -I${CEMU_SRC}/include \
//       -o mvcc_filter.so mvcc_filter.cc

#include <cstdint>
#include <cstring>

// cemu_csf.h is provided by the CEMU SDK inside the CEMU VM build environment.
#ifdef HAVE_CEMU_SDK
#include "cemu_csf.h"
#else
struct cemu_mr {
  void *addr;
  size_t len;
};
struct cemu_csf_args {
  struct cemu_mr mr[8];
  long long cparam1;
  long long cparam2;
  uint64_t ret;
};
#define CEMU_CSF_ENTRY(name) extern "C" void name
#endif

// ---------------------------------------------------------------------------
// Varint helpers — no RocksDB dependency in CSF
// ---------------------------------------------------------------------------

static const char *decode_varint32(const char *p, const char *limit,
                                   uint32_t *value) {
  uint32_t result = 0;
  for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
    uint32_t byte = static_cast<uint8_t>(*p++);
    if (byte & 0x80) {
      result |= ((byte & 0x7f) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return p;
    }
  }
  return nullptr;
}

static const char *decode_varint64(const char *p, const char *limit,
                                   uint64_t *value) {
  uint64_t result = 0;
  for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64_t byte = static_cast<uint8_t>(*p++);
    if (byte & 0x80) {
      result |= ((byte & 0x7f) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return p;
    }
  }
  return nullptr;
}

static inline uint64_t decode_fixed64_le(const char *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

static inline void write_uint32_le(char *dst, uint32_t v) {
  memcpy(dst, &v, 4);
}

// ---------------------------------------------------------------------------
// SST footer + block handle parsing
// ---------------------------------------------------------------------------

static const uint64_t kBlockBasedMagic = 0x88e241b785f4cff7ULL;
// New-format footer: 1-byte checksum + 2×BlockHandle + padding + 4-byte
// version + 8-byte magic = 53 bytes (Percona/RocksDB 8.x).
static const size_t kFooterSize = 53;

struct BlockHandle {
  uint64_t offset;
  uint64_t size;
};

static const char *decode_block_handle(const char *p, const char *limit,
                                       BlockHandle *bh) {
  p = decode_varint64(p, limit, &bh->offset);
  if (!p) return nullptr;
  return decode_varint64(p, limit, &bh->size);
}

// Parses the SST footer and returns the index block handle.
// Returns true on success, false if the file is too small or magic mismatches.
static bool parse_footer(const char *file_data, size_t file_len,
                         BlockHandle *index_bh) {
  if (file_len < kFooterSize) return false;

  const char *fp = file_data + file_len - kFooterSize;

  uint64_t magic;
  memcpy(&magic, fp + kFooterSize - 8, 8);
  if (magic != kBlockBasedMagic) return false;

  uint32_t version;
  memcpy(&version, fp + kFooterSize - 12, 4);

  // Version >= 1: first byte is the checksum type; version 0 has no prefix.
  const char *p = (version >= 1) ? fp + 1 : fp;
  const char *limit = fp + kFooterSize - 12;  // before version + magic

  // Skip metaindex block handle.
  BlockHandle meta_bh;
  p = decode_block_handle(p, limit, &meta_bh);
  if (!p) return false;

  p = decode_block_handle(p, limit, index_bh);
  return p != nullptr;
}

// ---------------------------------------------------------------------------
// Per-entry filter state (persists across data blocks for cross-block dedup)
// ---------------------------------------------------------------------------

struct FilterState {
  char key_buf[32768];     // reconstructed internal key (delta-decoding)
  size_t key_buf_len = 0;  // resets at each block boundary
  char prev_user_key[32760];
  size_t prev_user_key_len = 0;
  char *out_ptr;
  const char *out_end;
  uint64_t keys_seen = 0;
  bool overflow = false;
};

// ---------------------------------------------------------------------------
// Single data block processor
// ---------------------------------------------------------------------------

static void process_data_block(const char *block_data, size_t block_size,
                               uint64_t snapshot_seq, FilterState *st) {
  if (st->overflow || block_size < 4) return;

  // Each RocksDB data block ends with:
  //   [restart_offset_0..restart_offset_n-1] (n × 4 bytes)
  //   [num_restarts] (4 bytes)
  uint32_t num_restarts;
  memcpy(&num_restarts, block_data + block_size - 4, 4);
  if (num_restarts == 0) num_restarts = 1;
  const size_t restart_bytes = (1 + num_restarts) * 4;
  if (restart_bytes > block_size) return;
  const char *entry_end = block_data + block_size - restart_bytes;

  // Delta-encoding is block-local: reset key_buf_len at each block boundary.
  st->key_buf_len = 0;
  const char *p = block_data;

  while (p < entry_end && !st->overflow) {
    uint32_t shared, non_shared, val_len;

    p = decode_varint32(p, entry_end, &shared);
    if (!p) break;
    p = decode_varint32(p, entry_end, &non_shared);
    if (!p) break;
    p = decode_varint32(p, entry_end, &val_len);
    if (!p) break;

    if (static_cast<size_t>(entry_end - p) < non_shared + val_len) break;

    const size_t full_key_len = shared + non_shared;
    if (full_key_len > sizeof(st->key_buf) || shared > st->key_buf_len) break;
    memcpy(st->key_buf + shared, p, non_shared);
    st->key_buf_len = full_key_len;

    const char *val_ptr = p + non_shared;
    p = val_ptr + val_len;

    st->keys_seen++;

    // Internal key must carry the 8-byte seq+type footer.
    if (full_key_len < 8) continue;

    // Packed: (seq << 8) | type
    const uint64_t packed = decode_fixed64_le(st->key_buf + full_key_len - 8);
    const uint64_t seq = packed >> 8;
    const uint8_t type = static_cast<uint8_t>(packed & 0xff);

    // Rule 1: not visible to snapshot.
    if (seq > snapshot_seq) continue;

    const size_t user_key_len = full_key_len - 8;
    const char *user_key = st->key_buf;

    // Rule 2: duplicate — a newer visible version was already emitted for
    // this user_key (data blocks are sorted, so same user_key appears
    // consecutively in descending seq order).
    if (user_key_len == st->prev_user_key_len &&
        memcmp(user_key, st->prev_user_key, user_key_len) == 0) {
      continue;
    }

    // kTypeMerge (0x2): pass all operands through without updating
    // prev_user_key so the merge operator receives every operand.
    if (type != 0x2) {
      memcpy(st->prev_user_key, user_key, user_key_len);
      st->prev_user_key_len = user_key_len;
    }

    // Emit [uint32 key_len][key bytes][uint32 val_len][value bytes].
    const size_t needed = 4 + full_key_len + 4 + val_len;
    if (static_cast<size_t>(st->out_end - st->out_ptr) < needed) {
      st->overflow = true;
      break;
    }

    write_uint32_le(st->out_ptr, static_cast<uint32_t>(full_key_len));
    st->out_ptr += 4;
    memcpy(st->out_ptr, st->key_buf, full_key_len);
    st->out_ptr += full_key_len;
    write_uint32_le(st->out_ptr, static_cast<uint32_t>(val_len));
    st->out_ptr += 4;
    memcpy(st->out_ptr, val_ptr, val_len);
    st->out_ptr += val_len;
  }
}

// ---------------------------------------------------------------------------
// CSF entry point
// ---------------------------------------------------------------------------

CEMU_CSF_ENTRY(mvcc_filter)(struct cemu_csf_args *args) {
  const char *file_data = static_cast<const char *>(args->mr[0].addr);
  size_t file_len = args->mr[0].len;
  char *out_buf = static_cast<char *>(args->mr[1].addr);
  const size_t out_capacity = args->mr[1].len;
  const uint64_t snapshot_seq = static_cast<uint64_t>(args->cparam1);

  // Header occupies the first 16 bytes; KV stream follows.
  static const size_t kHeaderSize = 16;
  if (out_capacity < kHeaderSize) return;
  memset(out_buf, 0, kHeaderSize);

  // Locate the index block via the SST footer.
  BlockHandle index_bh;
  if (!parse_footer(file_data, file_len, &index_bh)) return;
  if (index_bh.offset + index_bh.size > file_len) return;

  const char *index_data = file_data + index_bh.offset;
  size_t index_size = index_bh.size;

  // Parse the index block to enumerate data block handles.
  // The index block uses the same delta-encoded format as data blocks.
  // Each entry value is a BlockHandle (two varint64s: offset, size).
  if (index_size < 4) return;
  uint32_t idx_num_restarts;
  memcpy(&idx_num_restarts, index_data + index_size - 4, 4);
  if (idx_num_restarts == 0) idx_num_restarts = 1;
  const size_t idx_restart_bytes = (1 + idx_num_restarts) * 4;
  if (idx_restart_bytes > index_size) return;
  const char *idx_entry_end = index_data + index_size - idx_restart_bytes;

  FilterState st{};
  st.out_ptr = out_buf + kHeaderSize;
  st.out_end = out_buf + out_capacity;

  // Key buffer for reconstructing index block keys (separators — not needed
  // for data, but required to advance delta-decoding state correctly).
  char idx_key_buf[32768];
  size_t idx_key_buf_len = 0;

  const char *p = index_data;
  while (p < idx_entry_end && !st.overflow) {
    uint32_t shared, non_shared, val_len;

    p = decode_varint32(p, idx_entry_end, &shared);
    if (!p) break;
    p = decode_varint32(p, idx_entry_end, &non_shared);
    if (!p) break;
    p = decode_varint32(p, idx_entry_end, &val_len);
    if (!p) break;

    if (static_cast<size_t>(idx_entry_end - p) < non_shared + val_len) break;

    const size_t full_key_len = shared + non_shared;
    if (full_key_len > sizeof(idx_key_buf) || shared > idx_key_buf_len) break;
    memcpy(idx_key_buf + shared, p, non_shared);
    idx_key_buf_len = full_key_len;

    const char *val_ptr = p + non_shared;
    p = val_ptr + val_len;

    BlockHandle data_bh;
    if (!decode_block_handle(val_ptr, val_ptr + val_len, &data_bh)) continue;

    // Block handle size does not include the 5-byte block trailer; the data
    // block content is exactly data_bh.size bytes starting at data_bh.offset.
    if (data_bh.offset + data_bh.size > file_len) continue;

    process_data_block(file_data + data_bh.offset, data_bh.size, snapshot_seq,
                       &st);
  }

  // Write the 16-byte header: result_bytes (8) + keys_seen (8).
  // IOCTL_CEMU_EXECUTE is _IOW so the ioctl struct is not returned; all
  // output must live in the FDM buffer that the host reads via pread().
  const uint64_t result_bytes =
      static_cast<uint64_t>(st.out_ptr - (out_buf + kHeaderSize));
  const uint64_t keys_seen = st.keys_seen;
  memcpy(out_buf,     &result_bytes, 8);
  memcpy(out_buf + 8, &keys_seen,    8);
}
