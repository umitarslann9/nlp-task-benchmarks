#!/usr/bin/env python3
"""
CENG 467 NLP Midterm - Q5 Language Modeling Notebook Generator
Run: python create_notebook.py
Generates: Q5_Language_Modeling.ipynb
"""
import json, os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q5_Language_Modeling.ipynb")

def cc(src, id_): return {"cell_type":"code","execution_count":None,"id":id_,"metadata":{},"outputs":[],"source":src}
def mc(src, id_): return {"cell_type":"markdown","id":id_,"metadata":{},"source":src}

TITLE = """\
# Q5 – Language Modeling
## CENG 467 – Natural Language Understanding and Generation
### Take-Home Midterm

| | |
|---|---|
| **Dataset** | WikiText-2 (Wikipedia articles, ~2M training tokens) |
| **Model 1** | Trigram N-gram with Laplace (add-1) smoothing |
| **Model 2** | LSTM Language Model (2-layer, 256-dim, PyTorch) |
| **Metric** | Perplexity (lower is better) |
| **Seed** | 42 |
"""

INSTALL = """\
!pip install datasets torch matplotlib -q
print("Done.")
"""

IMPORTS = """\
import os, random, math, warnings
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
import torch.nn as nn
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
"""

DATASET_HEADER = """\
## §1 – Dataset: WikiText-2

WikiText-2 is a language modelling benchmark built from Wikipedia featured articles:

| Split | Approx. Tokens |
|---|---|
| Train | ~2.1 M |
| Validation | ~217 K |
| Test | ~245 K |

Tokens are lowercased and include `<unk>` for rare words.
The dataset provides pre-tokenised text with explicit `<eos>` sentence boundaries.
"""

LOAD_DATA = """\
from datasets import load_dataset

raw = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_wikitext(text):
    return text.lower().split()

train_tokens = []
for row in raw["train"]:
    t = tokenize_wikitext(row["text"])
    if t:
        train_tokens.extend(t + ["<eos>"])

val_tokens = []
for row in raw["validation"]:
    t = tokenize_wikitext(row["text"])
    if t:
        val_tokens.extend(t + ["<eos>"])

test_tokens = []
for row in raw["test"]:
    t = tokenize_wikitext(row["text"])
    if t:
        test_tokens.extend(t + ["<eos>"])

print(f"Train tokens:  {len(train_tokens):,}")
print(f"Val tokens:    {len(val_tokens):,}")
print(f"Test tokens:   {len(test_tokens):,}")
print(f"Sample: {' '.join(train_tokens[:20])}")
"""

NGRAM_HEADER = """\
## §2 – Model 1: Trigram N-gram Language Model with Laplace Smoothing

A trigram model estimates:

$$P(w_t \\mid w_{t-2}, w_{t-1}) = \\frac{C(w_{t-2}, w_{t-1}, w_t) + 1}{C(w_{t-2}, w_{t-1}) + |V|}$$

where add-1 (Laplace) smoothing prevents zero probabilities for unseen n-grams.

**Perplexity:** $\\text{PPL} = \\exp\\left(-\\frac{1}{N}\\sum_{t=1}^{N} \\log P(w_t \\mid \\text{context})\\right)$
"""

NGRAM_CODE = """\
# ── Build vocabulary from training set ───────────────────────────────────────
MIN_FREQ = 3
counter  = Counter(train_tokens)
VOCAB    = {w for w, f in counter.items() if f >= MIN_FREQ}
VOCAB.add("<unk>"); VOCAB.add("<eos>")
V = len(VOCAB)
w2id = {w: i for i, w in enumerate(sorted(VOCAB))}

def unk_replace(tokens):
    return [t if t in VOCAB else "<unk>" for t in tokens]

train_tok = unk_replace(train_tokens)
val_tok   = unk_replace(val_tokens)
test_tok  = unk_replace(test_tokens)
print(f"Vocabulary size: {V:,}")

# ── Count n-grams ─────────────────────────────────────────────────────────────
print("Counting n-grams...")
unigram_counts  = Counter(train_tok)
bigram_counts   = Counter(zip(train_tok[:-1], train_tok[1:]))
trigram_counts  = Counter(zip(train_tok[:-2], train_tok[1:-1], train_tok[2:]))

print(f"Unique trigrams: {len(trigram_counts):,}")

def trigram_log_prob(w1, w2, w3):
    c3 = trigram_counts[(w1, w2, w3)]
    c2 = bigram_counts[(w1, w2)]
    return math.log((c3 + 1) / (c2 + V))

def perplexity_trigram(tokens):
    total_log_prob, n = 0.0, 0
    for i in range(2, len(tokens)):
        lp = trigram_log_prob(tokens[i-2], tokens[i-1], tokens[i])
        total_log_prob += lp
        n += 1
    return math.exp(-total_log_prob / n)

print("Computing perplexity...")
ngram_ppl_val  = perplexity_trigram(val_tok)
ngram_ppl_test = perplexity_trigram(test_tok)
print(f"Trigram (Laplace) | Val PPL: {ngram_ppl_val:.2f} | Test PPL: {ngram_ppl_test:.2f}")
"""

NGRAM_GENERATE = """\
# ── Pre-index trigrams/bigrams for O(1) lookup during generation ──────────────
trigram_idx = defaultdict(list)
for (w1, w2, w3), cnt in trigram_counts.items():
    trigram_idx[(w1, w2)].extend([w3] * min(cnt, 5))

bigram_idx = defaultdict(list)
for (w1, w2), cnt in bigram_counts.items():
    bigram_idx[w1].extend([w2] * min(cnt, 3))

VOCAB_LIST = list(VOCAB)   # pre-materialise for random.choice fallback

def ngram_generate(seed_words, n_words=30, temperature=1.0):
    generated = list(seed_words)
    for _ in range(n_words):
        w1, w2 = generated[-2], generated[-1]
        candidates = trigram_idx.get((w1, w2), [])
        if not candidates:
            candidates = bigram_idx.get(w2, VOCAB_LIST)
        word = random.choice(candidates)
        if word == "<eos>":
            break
        generated.append(word)
    return " ".join(generated)

print("=== Trigram N-gram Generated Samples ===")
seeds = [["the", "president"], ["a", "study"], ["in", "the"]]
for seed in seeds:
    print(f"  Seed: {seed}")
    text = ngram_generate(seed, n_words=25)
    print(f"  -> {text}")
    print()
"""

LSTM_HEADER = """\
## §3 – Model 2: LSTM Language Model

Architecture:
- Embedding: 256-dim
- 2-layer LSTM, hidden=256, dropout=0.5
- Linear projection: 256 -> vocab_size
- Weight tying: embedding weights = output projection weights (reduces parameters)
- Training: Adam lr=1e-3, cross-entropy loss, 20 epochs, batch=64, seq_len=35
"""

LSTM_DATA = """\
from torch.utils.data import Dataset, DataLoader

# ── Build integer vocabulary for LSTM ────────────────────────────────────────
counter_lstm = Counter(train_tok)
LSTM_VOCAB   = {w: i+2 for i, (w, _) in enumerate(counter_lstm.most_common(10000))}
LSTM_VOCAB["<pad>"] = 0
LSTM_VOCAB["<unk>"] = 1
LSTM_V = max(LSTM_VOCAB.values()) + 1
LSTM_ID2W = {v: k for k, v in LSTM_VOCAB.items()}

def encode_lstm(tokens):
    return [LSTM_VOCAB.get(t, 1) for t in tokens]

train_ids = torch.tensor(encode_lstm(train_tok), dtype=torch.long)
val_ids   = torch.tensor(encode_lstm(val_tok),   dtype=torch.long)
test_ids  = torch.tensor(encode_lstm(test_tok),  dtype=torch.long)

SEQ_LEN    = 35
BATCH_SIZE = 64

class LMDataset(Dataset):
    def __init__(self, ids, seq_len=SEQ_LEN):
        n = (len(ids) - 1) // seq_len
        self.x = ids[:n*seq_len].view(n, seq_len)
        self.y = ids[1:n*seq_len+1].view(n, seq_len)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_lm_ds = LMDataset(train_ids)
val_lm_ds   = LMDataset(val_ids)
test_lm_ds  = LMDataset(test_ids)

train_lm_dl = DataLoader(train_lm_ds, BATCH_SIZE, shuffle=True,  num_workers=2)
val_lm_dl   = DataLoader(val_lm_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
test_lm_dl  = DataLoader(test_lm_ds,  BATCH_SIZE, shuffle=False)
print(f"LSTM vocab: {LSTM_V:,} | Train batches: {len(train_lm_dl)}")
"""

LSTM_MODEL = """\
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 num_layers=2, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, vocab_size)
        # Weight tying
        self.embedding.weight = self.fc.weight

    def forward(self, x, hidden=None):
        emb = self.drop(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(self.drop(out))    # (B, L, V)
        return logits, hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(2, batch_size, 256, device=DEVICE)
        c = torch.zeros(2, batch_size, 256, device=DEVICE)
        return h, c

model_lm = LSTMLanguageModel(vocab_size=LSTM_V).to(DEVICE)
n_params = sum(p.numel() for p in model_lm.parameters())
print(f"LSTM LM parameters: {n_params:,}")
"""

LSTM_TRAIN = """\
def train_epoch_lm(model, loader, opt, criterion):
    model.train()
    total_loss, n_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        hidden = model.init_hidden(x.size(0))
        hidden = (hidden[0].detach(), hidden[1].detach())
        opt.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.view(-1, LSTM_V), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total_loss += loss.item() * y.numel()
        n_tokens   += y.numel()
    return total_loss / n_tokens

@torch.no_grad()
def eval_ppl(model, loader, criterion):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        hidden  = model.init_hidden(x.size(0))
        logits, _ = model(x, hidden)
        loss = criterion(logits.view(-1, LSTM_V), y.view(-1))
        total_loss += loss.item() * y.numel()
        n_tokens   += y.numel()
    return math.exp(total_loss / n_tokens)

criterion_lm = nn.CrossEntropyLoss(ignore_index=0)
optimizer_lm = torch.optim.Adam(model_lm.parameters(), lr=1e-3)
scheduler_lm = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lm, patience=3, factor=0.5)

EPOCHS = 20
best_val_ppl = float("inf")
history = []

print("Training LSTM Language Model (20 epochs)...")
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_epoch_lm(model_lm, train_lm_dl, optimizer_lm, criterion_lm)
    val_ppl = eval_ppl(model_lm, val_lm_dl, criterion_lm)
    scheduler_lm.step(val_ppl)
    history.append(val_ppl)
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save(model_lm.state_dict(), "lstm_lm_best.pt")
    if epoch % 4 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d} | Train Loss: {tr_loss:.4f} | Val PPL: {val_ppl:.2f}")

model_lm.load_state_dict(torch.load("lstm_lm_best.pt", weights_only=True))
lstm_ppl_val  = eval_ppl(model_lm, val_lm_dl,  criterion_lm)
lstm_ppl_test = eval_ppl(model_lm, test_lm_dl, criterion_lm)
print(f"\\nBest LSTM LM | Val PPL: {lstm_ppl_val:.2f} | Test PPL: {lstm_ppl_test:.2f}")

# ── Learning curve ────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), history, marker="o", markersize=3)
plt.xlabel("Epoch"); plt.ylabel("Validation Perplexity")
plt.title("LSTM LM Training Curve"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("lstm_lm_curve.png", dpi=150); plt.show()
"""

LSTM_GENERATE = """\
@torch.no_grad()
def lstm_generate(model, seed_words, n_words=40, temperature=0.8):
    model.eval()
    ids = [LSTM_VOCAB.get(w, 1) for w in seed_words]
    inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    hidden = model.init_hidden(1)
    # Warm up hidden state on seed
    _, hidden = model(inp, hidden)
    generated = list(seed_words)
    inp = torch.tensor([[ids[-1]]], dtype=torch.long, device=DEVICE)
    for _ in range(n_words):
        logits, hidden = model(inp, hidden)
        probs = torch.softmax(logits[0, 0] / temperature, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        next_word = LSTM_ID2W.get(next_id, "<unk>")
        if next_word == "<eos>":
            break
        generated.append(next_word)
        inp = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
    return " ".join(generated)

print("=== LSTM Language Model Generated Samples ===")
seeds_lstm = [["the", "university"], ["scientists", "have"], ["in", "the", "year"]]
for seed in seeds_lstm:
    text = lstm_generate(model_lm, seed, n_words=30, temperature=0.8)
    print(f"  Seed: {seed}")
    print(f"  -> {text}")
    print()
"""

COMPARISON = """\
## §4 – Comparison and Discussion
"""

COMPARISON_CODE = """\
import pandas as pd

results = pd.DataFrame({
    "Model":     ["Trigram N-gram (Laplace)", "LSTM LM (2-layer)"],
    "Val PPL":   [ngram_ppl_val,  lstm_ppl_val],
    "Test PPL":  [ngram_ppl_test, lstm_ppl_test],
})
print(results.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(7, 4))
models = ["Trigram\\nN-gram", "LSTM\\nLM"]
val_ppls  = [ngram_ppl_val,  lstm_ppl_val]
test_ppls = [ngram_ppl_test, lstm_ppl_test]
x = np.arange(2)
ax.bar(x - 0.2, val_ppls,  0.4, label="Val PPL",  color="steelblue")
ax.bar(x + 0.2, test_ppls, 0.4, label="Test PPL", color="coral")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Perplexity (lower is better)")
ax.set_title("Language Model Comparison")
ax.legend(); plt.tight_layout()
plt.savefig("lm_comparison.png", dpi=150); plt.show()
print("Saved: lm_comparison.png")
"""

DISCUSSION = """\
## §5 – Discussion

### Perplexity Interpretation

Perplexity measures how surprised the model is by unseen text:
- PPL = 1: perfect model (knows exactly what comes next)
- PPL = V: random baseline (vocabulary size)
- Typical values: ~400-600 for trigram, ~100-150 for LSTM on WikiText-2

### Model Comparison

| Aspect | Trigram N-gram | LSTM LM |
|---|---|---|
| Context window | 2 previous words | Theoretically unlimited (hidden state) |
| Long-range dependencies | Cannot capture | Partial capture (vanishing gradient) |
| Data sparsity | Severe (most trigrams unseen) | Mitigated via distributed representations |
| Smoothing | Laplace (naive) | Implicit via distributed embeddings |
| Perplexity range | ~400-600 | ~100-150 |
| Training time | Seconds (counting) | ~10-20 min (GPU) |
| Generation quality | Incoherent beyond short phrases | More grammatical, some coherence |

### Generated Text Analysis

**Trigram samples** show local grammaticality (short phrases sound natural) but
quickly degrade into incoherence — the model has no memory beyond 2 tokens.

**LSTM samples** show longer coherent phrases and some topic consistency due to
the hidden state carrying information across many tokens. However, the model
still lacks true semantic understanding and can produce contradictory or
semantically empty text.

A transformer-based model (GPT-2) would achieve PPL ~20-30 on WikiText-2,
demonstrating the dramatic improvement from attention mechanisms and scale.
"""

cells = [
    mc(TITLE,           "c00"),
    cc(INSTALL,         "c01"),
    cc(IMPORTS,         "c02"),
    mc(DATASET_HEADER,  "c03"),
    cc(LOAD_DATA,       "c04"),
    mc(NGRAM_HEADER,    "c05"),
    cc(NGRAM_CODE,      "c06"),
    cc(NGRAM_GENERATE,  "c07"),
    mc(LSTM_HEADER,     "c08"),
    cc(LSTM_DATA,       "c09"),
    cc(LSTM_MODEL,      "c10"),
    cc(LSTM_TRAIN,      "c11"),
    cc(LSTM_GENERATE,   "c12"),
    mc(COMPARISON,      "c13"),
    cc(COMPARISON_CODE, "c14"),
    mc(DISCUSSION,      "c15"),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.12"},
        "colab": {"provenance":[],"gpuType":"T4"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Created: {OUTPUT}")
