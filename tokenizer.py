import os
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict
import regex as re
import pickle
from utils import save_tokenizer

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
        ) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_freq_counts(
        input_path: str | os.PathLike, 
        chunk_start: int, 
        chunk_end: int, 
        special_tokens: list[str]
        ) -> dict[bytes, int]:
    
    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start)
    
    text = chunk.decode("utf-8")

    special_token_pattern = "|".join([re.escape(token) for token in special_tokens])
    docs = re.split(special_token_pattern, text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    freq = defaultdict(int)
    for doc in docs:
        for tok in re.finditer(PAT, doc):
            tok = tok.group()
            freq[tuple(list(tok.encode("utf-8")))] += 1

    return freq

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_freqs = defaultdict(int)
    num_chunks = os.cpu_count() * 2
    split_special_token = b'<|endoftext|>'

    with open(input_path, "rb") as f:
        chunks = find_chunk_boundaries(f, num_chunks, split_special_token)

    args = [(input_path, start, end, special_tokens) for start, end in zip(chunks, chunks[1:])]
        
    with Pool() as pool:
        res = pool.starmap(get_freq_counts, args)

    for freq in res:
        for word, counts in freq.items():
            word_freqs[word] += counts

    vocab = {i : bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")
    merges = []

    pair_to_words = defaultdict(set)
    pair_freqs = defaultdict(int)
    merged_words = []
    while len(vocab) < vocab_size:
        if not pair_freqs:
            for word, counts in word_freqs.items():
                for ch1, ch2 in zip(word, word[1:]):
                    pair_freqs[(ch1, ch2)] += counts
                    pair_to_words[(ch1, ch2)].add(word)
        else:
            for word in merged_words:
                for ch1, ch2 in zip(word, word[1:]):
                    pair_freqs[(ch1, ch2)] += word_freqs[word]
                    pair_to_words[(ch1, ch2)].add(word)

        max_freq = max(
            pair_freqs.items(), 
            key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]]))
        )[0]
        new_id = len(vocab)
        vocab[new_id] = vocab[max_freq[0]] + vocab[max_freq[1]]
        merges.append((vocab[max_freq[0]], vocab[max_freq[1]]))

        merged_words = []
        for word in list(pair_to_words[max_freq]):
            counts = word_freqs[word]
            new_word = []
            i = 0
            while i < len(word) - 1:
                if word[i] == max_freq[0] and word[i + 1] == max_freq[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            if i == len(word) - 1:
                new_word.append(word[i])

            new_word = tuple(new_word)
            del word_freqs[word]
            word_freqs[new_word] = counts

            merged_words.append(new_word)
            for ch1, ch2 in zip(word, word[1:]):
                pair_freqs[(ch1, ch2)] -= counts
                pair_to_words[(ch1, ch2)].discard(word) 

        del pair_freqs[(max_freq[0], max_freq[1])]

    return (vocab, merges)

if __name__ == "__main__":
    path = "sample-data/TinyStoriesV2-GPT4-valid.txt"
    (vocab, merges) = train_bpe(path, 1000, ['<|endoftext|>'])
    save_tokenizer(vocab, merges, "vocab.json", "merges.txt")
