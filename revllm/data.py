"""Tokenise FineWeb-Edu once into uint16 .bin files, then serve fixed (B, T) batches from a memmap."""
import os

import numpy as np
import torch

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Llama-2 SentencePiece vocab (32000), ungated
DATASET = ("HuggingFaceFW/fineweb-edu", "sample-10BT")


def prepare(out_dir, train_tokens=51_000_000, val_tokens=1_000_000, batch_docs=1000):
    """Stream FineWeb-Edu, tokenise, and write train.bin / val.bin (val = the first documents)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tok.eos_token_id
    ds = load_dataset(*DATASET, split="train", streaming=True)

    targets = [("val", val_tokens), ("train", train_tokens)]
    buf, docs = [], []
    it = iter(ds)
    for name, n_target in targets:
        path = os.path.join(out_dir, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_target,))
        filled = 0
        while filled < n_target:
            if not buf:
                docs = [next(it)["text"] for _ in range(batch_docs)]
                for ids in tok(docs, add_special_tokens=False)["input_ids"]:
                    buf.extend(ids)
                    buf.append(eos)
            take = min(len(buf), n_target - filled)
            arr[filled:filled + take] = buf[:take]
            buf = buf[take:]
            filled += take
            if name == "train" and filled % 5_000_000 < take:
                print(f"  train: {filled/1e6:.1f}M / {n_target/1e6:.0f}M tokens")
        arr.flush()
        print(f"wrote {path}: {n_target:,} tokens")
    return tok.vocab_size


class Batches:
    """Non-overlapping (T+1)-token windows, shuffled once with a fixed seed.

    Every run sees exactly the same windows in the same order regardless of batch size,
    so runs with different B are trained on identical data.
    """

    def __init__(self, path, seq_len, max_tokens=None, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        n = (len(self.data) - 1) // seq_len
        if max_tokens is not None:
            n = min(n, max_tokens // seq_len)
        self.T = seq_len
        self.order = np.random.default_rng(seed).permutation(n)

    def __len__(self):
        return len(self.order)

    def get(self, idx, device):
        rows = np.stack([self.data[i * self.T: i * self.T + self.T + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(rows)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x[:, :-1], x[:, 1:]

    def epoch(self, batch_size, device):
        for s in range(len(self.order) // batch_size):
            yield self.get(self.order[s * batch_size:(s + 1) * batch_size], device)
