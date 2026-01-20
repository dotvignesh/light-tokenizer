from utils import load_tokenizer
import regex as re
from collections.abc import Iterable, Iterator
import argparse

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None
                ):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else None

        self.vocab_byte_to_int = {self.vocab[i] : i for i in range(len(self.vocab))}

        self.merge_priority_by_id = {
            (self.vocab_byte_to_int[ch1], self.vocab_byte_to_int[ch2]): i 
            for i, (ch1, ch2) in enumerate(self.merges)
        }

        self.merge_result = {
            (self.vocab_byte_to_int[ch1], self.vocab_byte_to_int[ch2]): self.vocab_byte_to_int[ch1 + ch2]
            for ch1, ch2 in self.merges
        }

        if special_tokens:
            assert isinstance(special_tokens, list) and all(isinstance(t, str) for t in special_tokens), \
                "Special tokens should be a list of strings!"

            for tok in special_tokens:
                tok = tok.encode("utf-8")
                if tok not in self.vocab_byte_to_int:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = tok
                    self.vocab_byte_to_int[tok] = new_id
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        vocab, merges = load_tokenizer(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        
        ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if self.special_tokens:
            special_token_pattern = '(' + "|".join([re.escape(token) for token in self.special_tokens]) + ')'
            pieces = re.split(special_token_pattern, text)
        else:
            pieces = [text]

        for piece in pieces:
            if self.special_tokens and piece in self.special_tokens:
                ids.append(self.vocab_byte_to_int[piece.encode("utf-8")])
            else:
                for tok in re.finditer(PAT, piece):
                    tok = [self.vocab_byte_to_int[bytes([b])] for b in tok.group().encode("utf-8")]
                    while True:
                        temp = []
                        merge_pair, merge_priority = (), float("inf")
                        for (ch1, ch2) in zip(tok, tok[1:]):
                            if (ch1, ch2) in self.merge_priority_by_id:
                                if self.merge_priority_by_id[(ch1, ch2)] < merge_priority:
                                    merge_pair = (ch1, ch2)
                                    merge_priority = self.merge_priority_by_id[(ch1, ch2)]
                        
                        if not merge_pair: break

                        i = 0
                        while i < len(tok):
                            if i < len(tok) - 1 and tok[i] == merge_pair[0] and tok[i + 1] == merge_pair[1]:
                                temp.append(self.merge_result[merge_pair])
                                i += 2
                            else:
                                temp.append(tok[i])
                                i += 1

                        tok = temp

                    ids.extend(tok)

        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
            
    
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[id] for id in ids)
        return decoded_bytes.decode("utf-8", errors='replace')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode or decode text using BPE.")
    parser.add_argument("--encode", type=str, help="Text to encode")
    parser.add_argument("--decode", type=int, nargs="+", help="Token IDs to decode (space-separated integers)")
    parser.add_argument("--vocab", type=str, default="trained-tokenizers/TinyStories/vocab.json", help="Path to vocab.json")
    parser.add_argument("--merges", type=str, default="trained-tokenizers/TinyStories/merges.txt", help="Path to merges.txt")
    parser.add_argument("--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="Special tokens")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    if args.encode:
        ids = tokenizer.encode(args.encode)
        print(ids)
    
    if args.decode:
        text = tokenizer.decode(args.decode)
        print(text)