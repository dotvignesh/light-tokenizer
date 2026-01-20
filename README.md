# light-tokenizer

A parallelized BPE tokenizer built from scratch as part of Stanford's [CS336](https://stanford-cs336.github.io/spring2025/) assignment.

No HuggingFace. No SentencePiece. Just raw Python and a lot of profiling.

## What's here

- `train.py` - BPE training with multiprocessing for pre-tokenization
- `tokenizer.py` - CLI for BPE encoding and decoding
- `trained-tokenizers/` - Trained vocabulary and merge files for TinyStories (10K) and OpenWebText (32K)

## Quick start

```bash
# Train a tokenizer
python train.py --input sample-data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000

# Encode text
python tokenizer.py --encode "Hello world" --vocab trained-tokenizers/TinyStories/vocab.json --merges trained-tokenizers/TinyStories/merges.txt

# Decode tokens
python tokenizer.py --decode "15496 995" --vocab trained-tokenizers/TinyStories/vocab.json --merges trained-tokenizers/TinyStories/merges.txt
```

## Performance

Profiled with [Scalene](https://github.com/plasma-umass/scalene).

### Compression Ratios
Evaluated on validation sets:
- **OpenWebText (32K vocab)**: 4.37
- **TinyStories (10K vocab)**: 4.12

## Blog post

Wrote about the whole process here: [Building a BPE Tokenizer from Scratch](https://dotvignesh.substack.com/p/building-a-byte-pair-encoding-tokenizer)
