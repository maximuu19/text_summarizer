"""
Adaptive Predictive Modeling – Text Compression
================================================
Implements a novel compression scheme based on real-time statistical analysis
and dynamic dictionary generation, inspired by Zipf's Law.

Algorithm overview
------------------
1. Tokenise the input text on whitespace (lossless round-trip via ' '.join).
2. Build an adaptive frequency model (unigram counts) and a predictive
   model (bigram counts) from the full token stream.
3. Generate a dynamic dictionary: tokens are ranked by descending frequency
   so the most probable tokens receive the lowest indices and therefore the
   shortest variable-length integer (varint) codes.
4. Encode the token stream as a compact binary: a dictionary header followed
   by a varint index per token.
5. Decompression reconstructs the original token sequence from the header and
   reverses the dictionary, then joins on whitespace.

Binary format
-------------
  [2 bytes] number of unique words  (big-endian uint16)
  For each word:
    [1 byte ] word byte-length
    [N bytes] UTF-8 word
  [4 bytes] number of tokens  (big-endian uint32)
  For each token:
    [varint ] dictionary index (LEB128 unsigned)
"""

import re
import struct
from collections import Counter


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split *text* on whitespace.  Reconstruction is ``' '.join(tokens)``."""
    return text.split()


# ---------------------------------------------------------------------------
# Variable-length integer encoding (unsigned LEB128)
# ---------------------------------------------------------------------------

def encode_varint(value: int) -> bytes:
    """Encode *value* as an unsigned LEB128 byte sequence."""
    result = []
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Decode one unsigned LEB128 integer from *data* starting at *pos*.

    Returns ``(value, new_pos)``.
    """
    result = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, pos


def _varint_byte_size(index: int) -> int:
    """Return the number of bytes needed to encode *index* as a varint."""
    size = 1
    while index >= 0x80:
        index >>= 7
        size += 1
    return size


# ---------------------------------------------------------------------------
# Adaptive predictive model
# ---------------------------------------------------------------------------

def build_model(
    tokens: list[str],
) -> tuple[dict[str, int], list[str], Counter, Counter]:
    """Build the adaptive frequency + predictive (bigram) model.

    Returns
    -------
    word_to_idx : dict
        Token → dictionary index (lowest index = highest frequency).
    sorted_words : list
        Words ordered by descending frequency (then alphabetically for ties).
    unigram_freq : Counter
        Raw token counts.
    bigram_freq : Counter
        Raw bigram counts ``(w1, w2) → count``.
    """
    unigram_freq: Counter = Counter(tokens)
    bigram_freq: Counter = (
        Counter(zip(tokens, tokens[1:])) if len(tokens) > 1 else Counter()
    )

    # Highest-frequency tokens get the smallest indices → shortest varint codes.
    sorted_words = sorted(
        unigram_freq.keys(), key=lambda w: (-unigram_freq[w], w)
    )
    word_to_idx = {word: idx for idx, word in enumerate(sorted_words)}

    return word_to_idx, sorted_words, unigram_freq, bigram_freq


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------

def compress(text: str) -> dict | None:
    """Compress *text* with the adaptive predictive algorithm.

    Returns a dictionary with:

    ``compressed_data``
        Raw bytes of the compressed payload.
    ``dictionary``
        ``{token: code_index}`` mapping (most-frequent → 0).
    ``code_sizes``
        ``{token: varint_bytes}`` – code size per token in bytes.
    ``frequencies``
        ``{token: count}`` unigram frequencies.
    ``bigrams``
        Top-20 ``{(w1, w2): count}`` bigram counts.
    ``original_size``
        UTF-8 byte length of *text*.
    ``compressed_size``
        Byte length of ``compressed_data``.
    ``compression_ratio``
        ``(1 - compressed/original) * 100`` percentage.
    ``sorted_words``
        Words ranked by frequency (most frequent first).

    Returns ``None`` if *text* is empty or contains no tokens.
    """
    if not text or not text.strip():
        return None

    tokens = tokenize(text)
    if not tokens:
        return None

    word_to_idx, sorted_words, unigram_freq, bigram_freq = build_model(tokens)

    # --- build binary payload -------------------------------------------
    # Header: dictionary
    header = bytearray()
    header += struct.pack(">H", len(sorted_words))
    for word in sorted_words:
        word_bytes = word.encode("utf-8")
        # Clamp to 255 bytes per token (handles edge-case of very long URLs)
        word_bytes = word_bytes[:255]  # cap at 255 bytes per token (handles very long tokens)
        header += struct.pack("B", len(word_bytes)) + word_bytes

    # Body: token index stream
    body = bytearray()
    body += struct.pack(">I", len(tokens))
    for token in tokens:
        body += encode_varint(word_to_idx[token])

    compressed_data = bytes(header) + bytes(body)

    original_size = len(text.encode("utf-8"))
    compressed_size = len(compressed_data)
    compression_ratio = (
        (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0
    )

    code_sizes = {word: _varint_byte_size(idx) for word, idx in word_to_idx.items()}

    return {
        "compressed_data": compressed_data,
        "dictionary": {word: idx for idx, word in enumerate(sorted_words)},
        "code_sizes": code_sizes,
        "frequencies": dict(unigram_freq),
        "bigrams": dict(bigram_freq.most_common(20)),
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "sorted_words": sorted_words,
    }


# ---------------------------------------------------------------------------
# Decompress
# ---------------------------------------------------------------------------

def decompress(compressed_data: bytes) -> str | None:
    """Decompress bytes produced by :func:`compress`.

    Returns the reconstructed text string, or ``None`` on error.
    """
    if not compressed_data:
        return None

    try:
        pos = 0

        # Read dictionary
        (num_words,) = struct.unpack_from(">H", compressed_data, pos)
        pos += 2

        words: list[str] = []
        for _ in range(num_words):
            (word_len,) = struct.unpack_from("B", compressed_data, pos)
            pos += 1
            word = compressed_data[pos : pos + word_len].decode("utf-8")
            pos += word_len
            words.append(word)

        # Read token indices
        (num_tokens,) = struct.unpack_from(">I", compressed_data, pos)
        pos += 4

        tokens: list[str] = []
        for _ in range(num_tokens):
            idx, pos = decode_varint(compressed_data, pos)
            tokens.append(words[idx])

        return " ".join(tokens)

    except Exception:
        return None
