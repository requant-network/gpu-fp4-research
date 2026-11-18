# XBP1: original lossless XOR/bit-packing frame

This is an original implementation format for study 14, not a novelty claim.
All fields are little-endian. IEEE values are treated only as unsigned words;
there are no floating-point arithmetic or NaN-canonicalization operations.

## Header (32 bytes)

Python layout: `<4sBBBBQQQ`.

| Field | Meaning |
|---|---|
| magic | Four bytes `XBP1` |
| version | 1 |
| itemsize | 4 or 8 bytes per IEEE word |
| block_log2 | 4 through 12: 16 through 4096 words per block |
| flags | 0: block table plus packed payload; 1: whole-frame raw fallback |
| words | Number of original words, unsigned 64-bit |
| blocks | Number of table entries; zero for whole-frame raw fallback |
| bytes | Exact total encoded frame length, including this header |

Native single-frame implementations are limited to `words <= 2**32-1`. This is
an implementation limit, not a GPU hardware limit. Larger logical datasets are
streamed as multiple distinct frames. The complete frame, including its metadata,
is counted in encoded/transfer bytes. Raw fallback still has a 32-byte header.

## Whole-frame raw fallback

The header is followed by exactly `words * itemsize` unchanged bytes. Empty
inputs use this mode and consist of the header only. Trailing bytes are invalid.

## Block table (16 bytes per entry)

Python layout: `<QII`.

- `base`: first original unsigned IEEE word, stored in 64 bits. FP32 bases must
  have zero upper bits.
- `offset`: payload offset in 64-bit words, relative to the payload after the
  entire table. Offsets are contiguous and in block order.
- `code`: low 8 bits are packing width, next 16 bits are word count, top 8 bits
  are block mode (0 packed; 1 raw).

Every block except the last contains exactly `2**block_log2` words. The last
contains the remaining words. A packed block XORs every original word with its
base. Its width is the bit length of the OR of those differences. The resulting
unsigned differences are concatenated least-significant-bit first, then stored
as little-endian 64-bit words. Unused high bits are zero. Width zero needs no
payload and reconstructs a constant block.

A block uses raw mode when its padded packed payload would not be shorter than
its padded raw payload. A raw block has base zero, width `itemsize*8` and unchanged
word bytes padded with zero bytes to a 64-bit boundary. Whole-frame raw fallback
is selected when the complete table/payload representation is not shorter than
the raw frame. Consequently the emitted frame is at most original bytes plus
the 32-byte header; temporary candidate buffer capacity can be larger and must
be charged separately.

## Validation and safety

Before GPU submission the CPU validator checks the magic/version, supported
type/block/modes, exact total size, block counts, contiguous offsets, payload
extents, required raw/packed width rules and zero padding. Arbitrary malformed
lengths or offsets must not result in an out-of-bounds GPU read. Native GPU
range guards and an asynchronous error status provide additional protection
against modified device metadata; this is not a hardware-fault certificate.

The independent Python reference packs with arbitrary-precision integers and
unpacks by shifts/masks. Native CPU packing and CUDA extraction are separate
implementations and must produce the same bytes and exact original IEEE words.
The native API requires writable candidate/output/status buffers, non-overlapping
input/output/status storage, and caller-owned buffers alive until their CUDA
stream completes. CPU candidate buffers and device packets are 8-byte aligned.
These are caller buffer/lifetime preconditions, distinct from wire-format bounds
validation. Device status must be checked before using decoded results.
No compression or performance advantage is established by this format document.
