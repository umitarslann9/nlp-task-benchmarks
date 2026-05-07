#!/usr/bin/env python3
"""
CENG 467 NLP Midterm - Q4 Machine Translation Notebook Generator
Run: python create_notebook.py
Generates: Q4_Machine_Translation.ipynb
"""
import json, os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q4_Machine_Translation.ipynb")

def cc(src, id_): return {"cell_type":"code","execution_count":None,"id":id_,"metadata":{},"outputs":[],"source":src}
def mc(src, id_): return {"cell_type":"markdown","id":id_,"metadata":{},"source":src}

TITLE = """\
# Q4 – Machine Translation
## CENG 467 – Natural Language Understanding and Generation
### Take-Home Midterm

| | |
|---|---|
| **Dataset** | Multi30k EN->DE (30 K image-caption pairs) |
| **Model 1** | Seq2Seq + Bahdanau Attention (trained from scratch, PyTorch) |
| **Model 2** | Helsinki-NLP/opus-mt-en-de (pre-trained MarianMT transformer) |
| **Metrics** | BLEU · METEOR · ChrF · BERTScore |
| **Seed** | 42 |
"""

INSTALL = """\
!pip install datasets transformers torch sacrebleu bert-score sacremoses \\
             matplotlib sentencepiece nltk -q
import nltk
nltk.download("wordnet",   quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4",   quiet=True)
print("Done.")
"""

IMPORTS = """\
import os, random, warnings, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
"""

DATASET_HEADER = """\
## §1 – Dataset: Multi30k (EN -> DE)

Multi30k contains ~31,000 English-German sentence pairs from image captions
(Flickr30k). It is a standard benchmark for neural machine translation research.

| Split | Sentences |
|---|---|
| Train | 29,000 |
| Validation | 1,014 |
| Test | 1,000 |

Preprocessing steps:
1. Lowercase both languages
2. Basic tokenisation (split on whitespace + punctuation)
3. Build vocabulary from training set (min_freq = 2)
4. Pad sequences to batch max length, truncate at 50 tokens
"""

LOAD_DATA = """\
from datasets import load_dataset

raw = load_dataset("bentrevett/multi30k")
train_data = raw["train"]
val_data   = raw["validation"]
test_data  = raw["test"]

src_train = [ex["en"] for ex in train_data]
tgt_train = [ex["de"] for ex in train_data]
src_val   = [ex["en"] for ex in val_data]
tgt_val   = [ex["de"] for ex in val_data]
src_test  = [ex["en"] for ex in test_data]
tgt_test  = [ex["de"] for ex in test_data]

print(f"Train: {len(src_train)} | Val: {len(src_val)} | Test: {len(src_test)}")
print(f"\\nSample pair:")
print(f"  EN: {src_train[0]}")
print(f"  DE: {tgt_train[0]}")
"""

PREPROCESS = """\
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \\1 ", text)
    return text.split()

class Vocab:
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, sentences, min_freq=2):
        self.tok2id = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
        counter = Counter()
        for s in sentences:
            counter.update(tokenize(s))
        for w, f in counter.items():
            if f >= min_freq:
                self.tok2id[w] = len(self.tok2id)
        self.id2tok = {v:k for k,v in self.tok2id.items()}

    def encode(self, text, add_bos=False, add_eos=True, max_len=50):
        toks = tokenize(text)[:max_len]
        ids  = [self.tok2id.get(t, self.UNK) for t in toks]
        if add_bos: ids = [self.BOS] + ids
        if add_eos: ids = ids + [self.EOS]
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.id2tok.get(i, "<unk>")
            if tok in ("<pad>","<bos>","<eos>"): continue
            out.append(tok)
        return " ".join(out)

    def __len__(self): return len(self.tok2id)

src_vocab = Vocab(src_train, min_freq=2)
tgt_vocab = Vocab(tgt_train, min_freq=2)
print(f"Source vocab: {len(src_vocab):,} | Target vocab: {len(tgt_vocab):,}")

MAX_LEN    = 50
BATCH_SIZE = 128

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, srcs, tgts, sv, tv, max_len=MAX_LEN):
        self.pairs = []
        for s, t in zip(srcs, tgts):
            si = torch.tensor(sv.encode(s, max_len=max_len), dtype=torch.long)
            ti = torch.tensor(tv.encode(t, add_bos=True, max_len=max_len), dtype=torch.long)
            self.pairs.append((si, ti))

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=Vocab.PAD)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=Vocab.PAD)
    return src_pad, tgt_pad

train_ds  = TranslationDataset(src_train, tgt_train, src_vocab, tgt_vocab)
val_ds    = TranslationDataset(src_val,   tgt_val,   src_vocab, tgt_vocab)
test_ds   = TranslationDataset(src_test,  tgt_test,  src_vocab, tgt_vocab)
train_dl  = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=2)
val_dl    = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_dl   = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(f"Train batches: {len(train_dl)} | Val: {len(val_dl)} | Test: {len(test_dl)}")
"""

SEQ2SEQ_HEADER = """\
## §2 – Model 1: Seq2Seq with Bahdanau Attention (from scratch)

**Encoder:** BiGRU (2 layers, hidden=256)
**Attention:** Bahdanau additive attention — aligns each decoder step with encoder states
**Decoder:** GRU with attention context, projects to target vocabulary
**Training:** Adam lr=1e-3, label smoothing=0.1, teacher forcing=0.5, 15 epochs
"""

SEQ2SEQ_MODEL = """\
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers>1 else 0)
        self.fc  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.drop(self.embedding(x))               # (B, L, E)
        out, hidden = self.gru(emb)                      # out: (B,L,2H)  hidden: (2*layers,B,H)
        # Merge bidirectional -> (layers, B, H)
        hidden = torch.tanh(self.fc(
            torch.cat([hidden[-2], hidden[-1]], dim=1)   # last fwd+bwd
        )).unsqueeze(0).repeat(2, 1, 1)                  # (2, B, H) for 2-layer decoder
        return out, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=128):
        super().__init__()
        self.Wa = nn.Linear(enc_dim, attn_dim)
        self.Ua = nn.Linear(dec_dim, attn_dim)
        self.Va = nn.Linear(attn_dim, 1)

    def forward(self, decoder_hidden, encoder_out):
        # decoder_hidden: (1, B, dec_dim), encoder_out: (B, L, enc_dim)
        dh = decoder_hidden[-1].unsqueeze(1)            # (B, 1, dec_dim)
        scores = self.Va(torch.tanh(
            self.Wa(encoder_out) + self.Ua(dh)
        )).squeeze(2)                                    # (B, L)
        attn_weights = F.softmax(scores, dim=1)         # (B, L)
        context = (attn_weights.unsqueeze(2) * encoder_out).sum(1)  # (B, enc_dim)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256,
                 enc_dim=512, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention  = BahdanauAttention(enc_dim, hidden_dim)
        self.gru = nn.GRU(embed_dim + enc_dim, hidden_dim, num_layers,
                          batch_first=False, dropout=dropout if num_layers>1 else 0)
        self.fc_out = nn.Linear(hidden_dim + enc_dim + embed_dim, vocab_size)
        self.drop   = nn.Dropout(dropout)

    def forward(self, tgt_token, hidden, encoder_out):
        emb     = self.drop(self.embedding(tgt_token.unsqueeze(1)))  # (B,1,E)
        context, _ = self.attention(hidden, encoder_out)             # (B, enc_dim)
        rnn_in  = torch.cat([emb.squeeze(1), context], dim=1).unsqueeze(0)  # (1,B,E+enc)
        out, hidden = self.gru(rnn_in, hidden)                       # out:(1,B,H)
        pred = self.fc_out(torch.cat([
            out.squeeze(0), context, emb.squeeze(1)], dim=1))        # (B, vocab)
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad, tgt_bos, tgt_eos):
        super().__init__()
        self.encoder = encoder; self.decoder = decoder
        self.src_pad = src_pad; self.tgt_bos = tgt_bos; self.tgt_eos = tgt_eos

    def forward(self, src, tgt, teacher_forcing=0.5):
        enc_out, hidden = self.encoder(src)
        batch_size, tgt_len = tgt.shape[0], tgt.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs    = torch.zeros(tgt_len, batch_size, vocab_size, device=src.device)
        inp = tgt[:, 0]   # <BOS>
        for t in range(1, tgt_len):
            pred, hidden = self.decoder(inp, hidden, enc_out)
            outputs[t]   = pred
            use_tf = random.random() < teacher_forcing
            inp    = tgt[:, t] if use_tf else pred.argmax(1)
        return outputs

ENC_DIM = 512   # bidirectional GRU output dim
encoder  = Encoder(len(src_vocab)).to(DEVICE)
decoder  = Decoder(len(tgt_vocab), enc_dim=ENC_DIM).to(DEVICE)
model_s2s = Seq2Seq(encoder, decoder,
                     Vocab.PAD, Vocab.BOS, Vocab.EOS).to(DEVICE)

n_params = sum(p.numel() for p in model_s2s.parameters())
print(f"Seq2Seq+Attn parameters: {n_params:,}")
"""

SEQ2SEQ_TRAIN = """\
def train_epoch_s2s(model, loader, opt, criterion):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        opt.zero_grad()
        output = model(src, tgt)                 # (L, B, V)
        # Shift: predict tgt[1:] from output[1:]
        out = output[1:].reshape(-1, output.shape[-1])
        trg = tgt[:, 1:].reshape(-1)
        loss = criterion(out, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def translate_greedy(model, src_tensor, max_len=50):
    model.eval()
    enc_out, hidden = model.encoder(src_tensor.unsqueeze(0).to(DEVICE))
    inp = torch.tensor([Vocab.BOS], device=DEVICE)
    tokens = []
    for _ in range(max_len):
        pred, hidden = model.decoder(inp, hidden, enc_out)
        top1 = pred.argmax(1)
        if top1.item() == Vocab.EOS:
            break
        tokens.append(top1.item())
        inp = top1
    return tokens

criterion_s2s = nn.CrossEntropyLoss(ignore_index=Vocab.PAD, label_smoothing=0.1)
optimizer_s2s = torch.optim.Adam(model_s2s.parameters(), lr=1e-3)
scheduler_s2s = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_s2s, patience=2, factor=0.5)

EPOCHS = 15
best_loss = float("inf")
print("Training Seq2Seq+Attention (15 epochs)...")
for epoch in range(1, EPOCHS + 1):
    tr_loss = train_epoch_s2s(model_s2s, train_dl, optimizer_s2s, criterion_s2s)
    scheduler_s2s.step(tr_loss)
    if tr_loss < best_loss:
        best_loss = tr_loss
        torch.save(model_s2s.state_dict(), "seq2seq_best.pt")
    if epoch % 3 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d} | Train Loss: {tr_loss:.4f} | PPL: {math.exp(tr_loss):.2f}")

model_s2s.load_state_dict(torch.load("seq2seq_best.pt", weights_only=True))

# Translate test set
print("Translating test set...")
s2s_translations = []
for src_sentence in src_test:
    src_ids = torch.tensor(src_vocab.encode(src_sentence, max_len=MAX_LEN), dtype=torch.long)
    tok_ids = translate_greedy(model_s2s, src_ids)
    s2s_translations.append(tgt_vocab.decode(tok_ids))
print("Done.")
"""

MARIAN_HEADER = """\
## §3 – Model 2: Helsinki-NLP/opus-mt-en-de (Pre-trained MarianMT Transformer)

MarianMT is a Marian NMT model fine-tuned on OPUS multilingual corpora.
It uses a standard transformer encoder-decoder architecture.
We use it in **inference mode** (no fine-tuning on Multi30k).
"""

MARIAN_CODE = """\
from transformers import MarianMTModel, MarianTokenizer

print("Loading Helsinki-NLP/opus-mt-en-de ...")
marian_name = "Helsinki-NLP/opus-mt-en-de"
marian_tok  = MarianTokenizer.from_pretrained(marian_name)
marian_model = MarianMTModel.from_pretrained(marian_name).to(DEVICE)
marian_model.eval()

def marian_translate_batch(texts, batch_size=64):
    all_trans = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks  = marian_tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = marian_model.generate(**toks, num_beams=4, max_new_tokens=128)
        decoded = marian_tok.batch_decode(out, skip_special_tokens=True)
        all_trans.extend(decoded)
    return all_trans

print("Translating test set with MarianMT ...")
marian_translations = marian_translate_batch(src_test)
print(f"Done. {len(marian_translations)} translations generated.")
"""

EVAL_HEADER = """\
## §4 – Evaluation
"""

EVAL_CODE = """\
import sacrebleu
from bert_score import score as bert_score_fn

def compute_bleu(hyps, refs):
    return sacrebleu.corpus_bleu(hyps, [refs]).score

def compute_chrf(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs]).score

print("Computing BLEU...")
s2s_bleu  = compute_bleu(s2s_translations,  tgt_test)
mari_bleu = compute_bleu(marian_translations, tgt_test)

print("Computing ChrF...")
s2s_chrf  = compute_chrf(s2s_translations,    tgt_test)
mari_chrf = compute_chrf(marian_translations, tgt_test)

print("Computing BERTScore...")
try:
    _, _, s2s_bscore  = bert_score_fn(s2s_translations,    tgt_test,
                                       model_type="bert-base-multilingual-cased", verbose=False)
    _, _, mari_bscore = bert_score_fn(marian_translations, tgt_test,
                                       model_type="bert-base-multilingual-cased", verbose=False)
    s2s_bs  = s2s_bscore.mean().item()
    mari_bs = mari_bscore.mean().item()
except Exception as e:
    print(f"BERTScore failed: {e}, using 0.0")
    s2s_bs, mari_bs = 0.0, 0.0

# METEOR via NLTK
try:
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    def meteor_corpus(hyps, refs):
        scores = [meteor_score([word_tokenize(r.lower())], word_tokenize(h.lower()))
                  for h, r in zip(hyps, refs)]
        return np.mean(scores)
    s2s_meteor  = meteor_corpus(s2s_translations,    tgt_test)
    mari_meteor = meteor_corpus(marian_translations, tgt_test)
except Exception:
    s2s_meteor, mari_meteor = float("nan"), float("nan")

results = pd.DataFrame({
    "Model":      ["Seq2Seq + Bahdanau Attn", "MarianMT (Helsinki-NLP)"],
    "BLEU":       [s2s_bleu,  mari_bleu],
    "ChrF":       [s2s_chrf,  mari_chrf],
    "METEOR":     [s2s_meteor, mari_meteor],
    "BERTScore":  [s2s_bs,    mari_bs],
})
print("\\n=== Machine Translation Results ===")
print(results.to_string(index=False))

# ── Bar chart ─────────────────────────────────────────────────────────────────
metrics = ["BLEU", "ChrF", "BERTScore"]
s2s_vals  = [s2s_bleu,  s2s_chrf,  s2s_bs*100]
mari_vals = [mari_bleu, mari_chrf,  mari_bs*100]
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - 0.2, s2s_vals,  0.4, label="Seq2Seq+Attn", color="steelblue")
ax.bar(x + 0.2, mari_vals, 0.4, label="MarianMT",     color="coral")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("Score (0-100)"); ax.legend()
ax.set_title("MT: Seq2Seq vs MarianMT")
plt.tight_layout(); plt.savefig("mt_comparison.png", dpi=150); plt.show()
"""

QUALITATIVE_CODE = """\
## §5 – Qualitative Example
"""

QUALITATIVE = """\
print("=== Qualitative Translation Examples ===")
rng_qt = np.random.default_rng(SEED + 7)
sample_ids = rng_qt.choice(len(src_test), 3, replace=False)

for i, idx in enumerate(sample_ids, 1):
    print(f"\\n--- Example {i} ---")
    print(f"Source (EN):     {src_test[idx]}")
    print(f"Reference (DE):  {tgt_test[idx]}")
    print(f"Seq2Seq+Attn:   {s2s_translations[idx]}")
    print(f"MarianMT:        {marian_translations[idx]}")
"""

DISCUSSION = """\
## §6 – Discussion

### Metric Analysis

| Metric | What it measures | Limitation |
|---|---|---|
| **BLEU** | n-gram precision + brevity penalty | Prefers short outputs, ignores synonyms |
| **METEOR** | Precision/recall + stemming + synonyms | Slower, language-resource dependent |
| **ChrF** | Character-level n-gram F-score | Robust for morphologically rich languages (German) |
| **BERTScore** | Contextual semantic similarity | Computationally heavy, model-dependent |

### Architecture Trade-offs

| Aspect | Seq2Seq + Attention | MarianMT (Transformer) |
|---|---|---|
| Attention | Bahdanau (additive, O(n)) | Multi-head self-attention (O(n^2)) |
| Parallelism | Sequential (RNN bottleneck) | Fully parallel during training |
| Long-range | Struggles beyond ~30 tokens | Handles via positional encoding |
| Rare words | OOV tokens -> unk | BPE subword covers everything |
| Training data | Multi30k 29K pairs | OPUS (~60M pairs) |

MarianMT's large pre-training advantage explains the performance gap.
The Seq2Seq + Attention model learns reasonable translations from 29K pairs but
struggles with compound German words and long-range agreement.
"""

cells = [
    mc(TITLE,            "c00"),
    cc(INSTALL,          "c01"),
    cc(IMPORTS,          "c02"),
    mc(DATASET_HEADER,   "c03"),
    cc(LOAD_DATA,        "c04"),
    cc(PREPROCESS,       "c05"),
    mc(SEQ2SEQ_HEADER,   "c06"),
    cc(SEQ2SEQ_MODEL,    "c07"),
    cc(SEQ2SEQ_TRAIN,    "c08"),
    mc(MARIAN_HEADER,    "c09"),
    cc(MARIAN_CODE,      "c10"),
    mc(EVAL_HEADER,      "c11"),
    cc(EVAL_CODE,        "c12"),
    mc(QUALITATIVE_CODE, "c13"),
    cc(QUALITATIVE,      "c14"),
    mc(DISCUSSION,       "c15"),
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
