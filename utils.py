from __future__ import annotations
import json
from functools import lru_cache

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    byte_encoder = gpt2_bytes_to_unicode()  # byte (int) -> printable char
    
    def encode_bytes(b):
        """Convert bytes to GPT-2 printable string"""
        return ''.join(byte_encoder[byte] for byte in b)
    
    # Save vocab: {"encoded_string": token_id, ...}
    vocab_serializable = {
        encode_bytes(token_bytes): token_id 
        for token_id, token_bytes in vocab.items()
    }
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_serializable, f, ensure_ascii=False)
    
    # Save merges: one per line, space-separated
    with open(merges_path, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            f.write(f"{encode_bytes(token1)} {encode_bytes(token2)}\n")

def load_tokenizer(vocab_path, merges_path):
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}  # printable char -> byte (int)
    
    def decode_string(s):
        """Convert GPT-2 printable string back to bytes"""
        return bytes([byte_decoder[c] for c in s])
    
    # Load vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_loaded = json.load(f)
    vocab = {token_id: decode_string(s) for s, token_id in vocab_loaded.items()}
    
    # Load merges
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges = [
            (decode_string(parts[0]), decode_string(parts[1]))
            for line in f
            for parts in [line.rstrip().split(' ')]
        ]
    
    return vocab, merges