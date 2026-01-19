# light-tokenizer

A parallelized BPE tokenizer built from scratch as part of Stanford's [CS336](https://stanford-cs336.github.io/spring2025/) assignment.

No HuggingFace. No SentencePiece. Just raw Python and a lot of profiling.

## What's here

- `train.py` - BPE training with multiprocessing for pre-tokenization
- `tokenizer.py` - Encode/decode implementation (Coming soon)
- `vocab/` - Trained vocabulary and merge files for TinyStories-GPT4 (10K vocab)

## Quick start

```bash
# Train a tokenizer
python train.py --input sample-data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000

# Encode text (Coming soon)
# python tokenizer.py --encode "Hello world"
```

## Performance

~2.7x speedup achieved through:
- Parallelized pre-tokenization using `multiprocessing`
- Caching byte pair locations to avoid redundant merge scans

Profiled with [Scalene](https://github.com/plasma-umass/scalene).

## Blog post

Wrote about the whole process here: [Building a BPE Tokenizer from Scratch](https://dotvignesh.substack.com/p/building-a-byte-pair-encoding-tokenizer)

## Coming soon

- `tokenizer.py` implementation for encoding and decoding
- OpenWebText 32K vocabulary GPT-2 style tokenizer
