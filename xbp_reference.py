"""Independent Python XBP1 format reference using arbitrary-precision bit packing."""
import struct
import numpy as np

HEADER = struct.Struct('<4sBBBBQQQ')
ENTRY = struct.Struct('<QII')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _block_log(block_words):
    _require(isinstance(block_words, int) and 16 <= block_words <= 4096 and
             block_words & (block_words - 1) == 0, 'Expected power-of-two block size from 16 to 4096')
    return block_words.bit_length() - 1


def encode_reference(original, block_words=256):
    _require(isinstance(original, np.ndarray) and original.flags.c_contiguous, 'Expected contiguous ndarray')
    _require(original.dtype in (np.dtype('<f4'), np.dtype('<f8')), 'Expected little-endian FP32 or FP64')
    log = _block_log(block_words)
    itemsize = original.dtype.itemsize
    word_type = np.dtype('<u4' if itemsize == 4 else '<u8')
    words = original.view(word_type).reshape(-1)
    raw = words.tobytes()
    rows, payload = [], bytearray()
    for begin in range(0, words.size, block_words):
        block = words[begin:begin + block_words]
        base = int(block[0]); union = 0
        differences = [int(word) ^ base for word in block]
        for difference in differences:
            union |= difference
        width = union.bit_length()
        packed_words = (len(block) * width + 63) // 64
        raw_words = (block.nbytes + 7) // 8
        offset = len(payload) // 8
        if packed_words < raw_words:
            packed = 0
            for index, difference in enumerate(differences):
                packed |= difference << (index * width)
            encoded = packed.to_bytes(packed_words * 8, 'little')
            mode = 0
        else:
            base, width, mode = 0, itemsize * 8, 1
            encoded = block.tobytes().ljust(raw_words * 8, b'\x00')
        rows.append(ENTRY.pack(base, offset, width | (len(block) << 8) | (mode << 24)))
        payload.extend(encoded)
    total = HEADER.size + len(rows) * ENTRY.size + len(payload)
    if total >= HEADER.size + len(raw):
        return HEADER.pack(b'XBP1', 1, itemsize, log, 1, words.size, 0, HEADER.size + len(raw)) + raw
    return (HEADER.pack(b'XBP1', 1, itemsize, log, 0, words.size, len(rows), total) +
            b''.join(rows) + payload)


def validate_packet(packet):
    packet = bytes(packet)
    _require(len(packet) >= HEADER.size, 'Truncated XBP1 header')
    magic, version, itemsize, log, flags, words, blocks, total = HEADER.unpack_from(packet)
    _require(magic == b'XBP1' and version == 1, 'Unknown XBP1 magic/version')
    _require(itemsize in (4, 8) and 4 <= log <= 12 and flags in (0, 1), 'Invalid XBP1 type/block/mode')
    _require(total == len(packet), 'XBP1 total size does not match exact packet bytes')
    block_words = 1 << log
    base = {'words': words, 'itemsize': itemsize, 'dtype': '<f4' if itemsize == 4 else '<f8',
            'block_words': block_words, 'block_log2': log, 'total_bytes': total, 'blocks': blocks}
    if flags == 1:
        _require(blocks == 0 and total == HEADER.size + words * itemsize, 'Invalid XBP1 raw extent')
        return {**base, 'mode': 'raw', 'payload_begin': HEADER.size, 'entries': []}
    _require(words > 0 and blocks == (words + block_words - 1) // block_words, 'Invalid XBP1 block count')
    payload_begin = HEADER.size + blocks * ENTRY.size
    _require(payload_begin <= total and (total - payload_begin) % 8 == 0, 'Truncated XBP1 block table/payload')
    _require(total < HEADER.size + words * itemsize, 'Non-beneficial packed XBP1 frame')
    cursor = 0; entries = []
    for index in range(blocks):
        first, offset, code = ENTRY.unpack_from(packet, HEADER.size + index * ENTRY.size)
        width, count, mode = code & 255, (code >> 8) & 65535, code >> 24
        expected_count = min(block_words, words - index * block_words)
        _require(count == expected_count and offset == cursor, 'Invalid XBP1 block extent/offset')
        _require(mode in (0, 1) and first < (1 << (itemsize * 8)), 'Invalid XBP1 block mode/base')
        raw_words = (count * itemsize + 7) // 8
        if mode == 0:
            _require(width < itemsize * 8, 'Invalid XBP1 packed width')
            length = (count * width + 63) // 64
            _require(length < raw_words, 'Non-beneficial packed XBP1 block')
        else:
            _require(width == itemsize * 8 and first == 0, 'Invalid XBP1 raw block')
            length = raw_words
        start, stop = payload_begin + offset * 8, payload_begin + (offset + length) * 8
        _require(stop <= total, 'XBP1 block payload extends beyond packet')
        if mode == 0 and length and (count * width) % 64:
            trailing = int.from_bytes(packet[stop - 8:stop], 'little')
            _require(trailing >> ((count * width) % 64) == 0, 'Nonzero XBP1 unused packed bits')
        if mode == 1 and count * itemsize % 8:
            _require(not any(packet[start + count * itemsize:stop]), 'Nonzero XBP1 raw padding')
        entries.append((first, offset, width, count, mode, length))
        cursor += length
    _require(payload_begin + cursor * 8 == total, 'Trailing XBP1 payload words')
    return {**base, 'mode': 'packed', 'payload_begin': payload_begin, 'entries': entries}


def decode_reference(packet):
    packet = bytes(packet)
    meta = validate_packet(packet)
    if meta['mode'] == 'raw':
        return np.frombuffer(packet, meta['dtype'], offset=HEADER.size).copy()
    words = []
    for first, offset, width, count, mode, length in meta['entries']:
        begin = meta['payload_begin'] + offset * 8
        raw = packet[begin:begin + length * 8]
        if mode:
            block = [int.from_bytes(raw[i * meta['itemsize']:(i + 1) * meta['itemsize']], 'little')
                     for i in range(count)]
        else:
            packed = int.from_bytes(raw, 'little')
            mask = (1 << width) - 1
            block = [first ^ ((packed >> (i * width)) & mask) for i in range(count)]
        words.extend(block)
    word_type = np.dtype('<u4' if meta['itemsize'] == 4 else '<u8')
    return np.asarray(words, word_type).view(meta['dtype'])
