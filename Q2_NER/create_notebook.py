#!/usr/bin/env python3
"""
CENG 467 NLP Midterm - Q2 NER Notebook Generator
Run: python create_notebook.py
Generates: Q2_NER.ipynb
"""
import json, os

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Q2_NER.ipynb")

def cc(src, id_): return {"cell_type":"code","execution_count":None,"id":id_,"metadata":{},"outputs":[],"source":src}
def mc(src, id_): return {"cell_type":"markdown","id":id_,"metadata":{},"source":src}

# ══════════════════════════════════════════════════════════════════════════════
TITLE = """\
# Q2 – Named Entity Recognition
## CENG 467 – Natural Language Understanding and Generation
### Take-Home Midterm

| | |
|---|---|
| **Dataset** | CoNLL-2003 (English NER, 4 entity types) |
| **Models** | BiLSTM-CRF &nbsp;·&nbsp; BERT fine-tuned for token classification |
| **Evaluation** | Precision / Recall / F1 (entity-level, seqeval) |
| **Random Seed** | 42 |
"""

INSTALL = """\
!pip install "datasets<3.0.0" transformers torch scikit-learn seqeval pytorch-crf \\
             matplotlib seaborn -q
print("Done.")
"""

IMPORTS = """\
import os, random, warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
"""

DATASET_HEADER = """\
## §1 – Dataset: CoNLL-2003

CoNLL-2003 is the standard benchmark for English Named Entity Recognition,
annotated with four entity types using **BIO tagging**:

| Tag prefix | Entity type |
|---|---|
| B- / I- | **PER** – persons |
| B- / I- | **ORG** – organisations |
| B- / I- | **LOC** – locations |
| B- / I- | **MISC** – miscellaneous named entities |
| O | Outside any entity |

**BIO scheme:** `B-XXX` marks the beginning of an entity, `I-XXX` continues it,
`O` marks non-entity tokens. The full tag set has 9 labels (O + 4×B + 4×I).
"""

LOAD_DATA = """\
from datasets import load_dataset

raw = load_dataset("conll2003", trust_remote_code=True)
train_ds = raw["train"]
val_ds   = raw["validation"]
test_ds  = raw["test"]

# Tag names: index -> string
TAG_NAMES = train_ds.features["ner_tags"].feature.names
ID2TAG    = {i: t for i, t in enumerate(TAG_NAMES)}
TAG2ID    = {t: i for i, t in ID2TAG.items()}
NUM_TAGS  = len(TAG_NAMES)

print("Tag set:", TAG_NAMES)
print(f"Num tags: {NUM_TAGS}")
print(f"Train: {len(train_ds)} sentences | Val: {len(val_ds)} | Test: {len(test_ds)}")

# Show one example
ex = train_ds[0]
print("\\nExample sentence:")
for tok, tag_id in zip(ex["tokens"], ex["ner_tags"]):
    print(f"  {tok:<20} {ID2TAG[tag_id]}")
"""

DATA_STATS = """\
# ── Entity distribution ───────────────────────────────────────────────────────
tag_counts = Counter()
for ex in train_ds:
    for t in ex["ner_tags"]:
        tag_counts[ID2TAG[t]] += 1

labels = list(tag_counts.keys())
counts = [tag_counts[l] for l in labels]

plt.figure(figsize=(10, 4))
colors = ["#aaaaaa" if l == "O" else "#2196F3" for l in labels]
plt.bar(labels, counts, color=colors)
plt.title("CoNLL-2003 Train: Tag Frequency Distribution")
plt.xlabel("BIO Tag"); plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("conll_tag_dist.png", dpi=150)
plt.show()
print("Saved: conll_tag_dist.png")

# Sentence length stats
sent_lens = [len(ex["tokens"]) for ex in train_ds]
print(f"\\nAvg sentence length: {np.mean(sent_lens):.1f} tokens")
print(f"Max sentence length: {max(sent_lens)} tokens")
"""

BILSTM_CRF_HEADER = """\
## §2 – Model 1: BiLSTM-CRF (Classical / Hybrid)

Architecture:
- Word embeddings (100-dim, randomly initialised)
- 2-layer bidirectional LSTM (hidden=128 per direction → 256 concat)
- Linear projection to tag space (256 → 9)
- CRF decoding layer (`pytorch-crf`) for globally consistent tag sequences

The CRF layer learns transition probabilities between tags, preventing invalid
sequences (e.g., `I-PER` following `B-ORG`).
"""

BILSTM_CRF_DATA = """\
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

PAD_TAG_ID = TAG2ID["O"]   # pad with O tag
MAX_LEN    = 128
BATCH_SIZE = 32

# ── Vocabulary ────────────────────────────────────────────────────────────────
def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for ex in dataset:
        counter.update([t.lower() for t in ex["tokens"]])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

VOCAB = build_vocab(train_ds)
print(f"Vocabulary size: {len(VOCAB):,}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, hf_ds, vocab, max_len=MAX_LEN):
        self.items = []
        for ex in hf_ds:
            toks = [vocab.get(t.lower(), 1) for t in ex["tokens"]][:max_len]
            tags = ex["ner_tags"][:max_len]
            pad  = max_len - len(toks)
            mask = [True] * len(toks) + [False] * pad
            toks = toks + [0] * pad
            tags = tags + [PAD_TAG_ID] * pad
            self.items.append((
                torch.tensor(toks, dtype=torch.long),
                torch.tensor(tags, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
            ))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

train_ner = NERDataset(train_ds, VOCAB)
val_ner   = NERDataset(val_ds,   VOCAB)
test_ner  = NERDataset(test_ds,  VOCAB)

pin = torch.cuda.is_available()
train_dl = DataLoader(train_ner, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)
val_dl   = DataLoader(val_ner,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
test_dl  = DataLoader(test_ner,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
print(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")
"""

BILSTM_CRF_MODEL = """\
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=100,
                 hidden_dim=128, num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop     = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim * 2, num_tags)
        self.crf      = CRF(num_tags, batch_first=True)

    def emissions(self, x):
        emb = self.drop(self.embedding(x))
        out, _ = self.lstm(emb)
        return self.fc(self.drop(out))           # (B, L, num_tags)

    def forward(self, x, tags, mask):
        emits = self.emissions(x)
        return -self.crf(emits, tags, mask=mask, reduction="mean")

    def decode(self, x, mask):
        emits = self.emissions(x)
        return self.crf.decode(emits, mask=mask) # list of lists

model_crf = BiLSTMCRF(vocab_size=len(VOCAB), num_tags=NUM_TAGS).to(DEVICE)
print(f"BiLSTM-CRF parameters: {sum(p.numel() for p in model_crf.parameters()):,}")
"""

BILSTM_CRF_TRAIN = """\
from seqeval.metrics import precision_score, recall_score, f1_score as seq_f1
from seqeval.metrics import classification_report as seq_report

def ids_to_labels(id_seq, mask_seq):
    return [ID2TAG[i] for i, m in zip(id_seq, mask_seq) if m]

def evaluate_crf(model, loader):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for toks, tags, mask in loader:
            toks, tags, mask = toks.to(DEVICE), tags.to(DEVICE), mask.to(DEVICE)
            pred_ids = model.decode(toks, mask)
            for i in range(len(toks)):
                m = mask[i].cpu().tolist()
                true_seq = ids_to_labels(tags[i].cpu().tolist(), m)
                pred_seq = ids_to_labels(pred_ids[i] + [0]*(sum(m)-len(pred_ids[i])), m)
                all_true.append(true_seq)
                all_pred.append(pred_seq)
    p  = precision_score(all_true, all_pred)
    r  = recall_score(all_true, all_pred)
    f1 = seq_f1(all_true, all_pred)
    return p, r, f1, all_true, all_pred

optimizer_crf = torch.optim.Adam(model_crf.parameters(), lr=1e-3)
scheduler_crf = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_crf, patience=2, factor=0.5)

EPOCHS = 10
best_f1_crf = 0.0
print("Training BiLSTM-CRF (10 epochs)...")
for epoch in range(1, EPOCHS + 1):
    model_crf.train()
    total_loss = 0.0
    for toks, tags, mask in train_dl:
        toks, tags, mask = toks.to(DEVICE), tags.to(DEVICE), mask.to(DEVICE)
        optimizer_crf.zero_grad()
        loss = model_crf(toks, tags, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model_crf.parameters(), 5.0)
        optimizer_crf.step()
        total_loss += loss.item()
    val_p, val_r, val_f1, _, _ = evaluate_crf(model_crf, val_dl)
    scheduler_crf.step(1 - val_f1)
    if val_f1 > best_f1_crf:
        best_f1_crf = val_f1
        torch.save(model_crf.state_dict(), "bilstm_crf_best.pt")
    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:2d} | Loss: {total_loss/len(train_dl):.4f} | "
              f"Val P: {val_p:.4f} | R: {val_r:.4f} | F1: {val_f1:.4f}")

model_crf.load_state_dict(torch.load("bilstm_crf_best.pt", weights_only=True))
crf_p, crf_r, crf_f1, crf_true, crf_pred = evaluate_crf(model_crf, test_dl)
print(f"\\nBiLSTM-CRF Test  =>  P: {crf_p:.4f}  R: {crf_r:.4f}  F1: {crf_f1:.4f}")
print(seq_report(crf_true, crf_pred))
"""

BERT_NER_HEADER = """\
## §3 – Model 2: BERT Fine-tuned for Token Classification (Contextual)

- Base model: `bert-base-cased` (important: cased for NER, capitalisation is a strong signal)
- Tokeniser: WordPiece with subword alignment — only the **first subword** of each word
  receives the gold label; continuation subwords are masked with `-100`
- Head: Linear(768 → 9) on top of the last hidden state
- Training: HuggingFace Trainer, lr=2e-5, 3 epochs, batch=16
"""

BERT_NER_DATA = """\
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset as HFDataset

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align(examples):
    tokenized = tokenizer_bert(
        examples["tokens"],
        truncation=True,
        max_length=128,
        is_split_into_words=True,
    )
    labels_out = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids   = tokenized.word_ids(batch_index=i)
        label_seq, prev_word = [], None
        for wid in word_ids:
            if wid is None:
                label_seq.append(-100)           # [CLS] / [SEP]
            elif wid != prev_word:
                label_seq.append(label_ids[wid]) # first subword -> real label
            else:
                label_seq.append(-100)           # continuation subword -> ignore
            prev_word = wid
        labels_out.append(label_seq)
    tokenized["labels"] = labels_out
    return tokenized

print("Tokenising datasets for BERT...")
cols_to_remove = ["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"]
hf_train = raw["train"].map(tokenize_and_align, batched=True, remove_columns=cols_to_remove)
hf_val   = raw["validation"].map(tokenize_and_align, batched=True, remove_columns=cols_to_remove)
hf_test  = raw["test"].map(tokenize_and_align, batched=True, remove_columns=cols_to_remove)

print(hf_train)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_bert)
"""

BERT_NER_TRAIN = """\
from seqeval.metrics import (precision_score, recall_score,
                              f1_score as seq_f1,
                              classification_report as seq_report)

id2tag_bert = ID2TAG.copy()
tag2id_bert = TAG2ID.copy()

def compute_metrics_ner(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = [], []
    for pred_row, lbl_row in zip(preds, labels):
        true_seq, pred_seq = [], []
        for p, l in zip(pred_row, lbl_row):
            if l != -100:
                true_seq.append(ID2TAG[l])
                pred_seq.append(ID2TAG[p])
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)
    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
        "f1":        seq_f1(true_seqs, pred_seqs),
    }

model_bert_ner = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=NUM_TAGS,
    id2label=id2tag_bert,
    label2id=tag2id_bert,
)

training_args_ner = TrainingArguments(
    output_dir                  = "./bert_ner_ckpt",
    num_train_epochs            = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate               = 2e-5,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    fp16                        = torch.cuda.is_available(),
    seed                        = SEED,
    report_to                   = "none",
    logging_steps               = 100,
)

trainer_ner = Trainer(
    model            = model_bert_ner,
    args             = training_args_ner,
    train_dataset    = hf_train,
    eval_dataset     = hf_val,
    processing_class = tokenizer_bert,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics_ner,
)

print("Fine-tuning BERT for NER (~15 min on T4 GPU) ...")
trainer_ner.train()
"""

BERT_NER_EVAL = """\
res_bert = trainer_ner.evaluate(hf_test)
bert_p   = res_bert["eval_precision"]
bert_r   = res_bert["eval_recall"]
bert_f1  = res_bert["eval_f1"]
print(f"BERT NER Test  =>  P: {bert_p:.4f}  R: {bert_r:.4f}  F1: {bert_f1:.4f}")

# Detailed predictions for error analysis
pred_out_bert = trainer_ner.predict(hf_test)
bert_preds_raw = np.argmax(pred_out_bert.predictions, axis=-1)

# Convert back to label sequences
bert_true_seqs, bert_pred_seqs = [], []
for pred_row, lbl_row in zip(bert_preds_raw, pred_out_bert.label_ids):
    ts, ps = [], []
    for p, l in zip(pred_row, lbl_row):
        if l != -100:
            ts.append(ID2TAG[int(l)])
            ps.append(ID2TAG[int(p)])
    bert_true_seqs.append(ts)
    bert_pred_seqs.append(ps)

print()
print(seq_report(bert_true_seqs, bert_pred_seqs))
"""

COMPARISON_HEADER = """\
## §4 – Results Comparison
"""

COMPARISON = """\
import pandas as pd

results = pd.DataFrame({
    "Model":      ["BiLSTM-CRF", "BERT-base-cased"],
    "Approach":   ["Classical/Hybrid (CRF decoding)", "Contextual (transformer)"],
    "Precision":  [crf_p, bert_p],
    "Recall":     [crf_r, bert_r],
    "F1":         [crf_f1, bert_f1],
})
print(results.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(2)
w = 0.25
ax.bar(x - w,   results["Precision"], w, label="Precision", color="steelblue")
ax.bar(x,       results["Recall"],    w, label="Recall",    color="coral")
ax.bar(x + w,   results["F1"],        w, label="F1",        color="green")
ax.set_xticks(x); ax.set_xticklabels(["BiLSTM-CRF", "BERT"])
ax.set_ylim(0, 1.0); ax.set_ylabel("Score"); ax.legend()
ax.set_title("NER Model Comparison (CoNLL-2003 Test Set)")
plt.tight_layout(); plt.savefig("ner_comparison.png", dpi=150); plt.show()
"""

ERROR_ANALYSIS_HEADER = """\
## §5 – Error Analysis
"""

ERROR_ANALYSIS = """\
print("=== BiLSTM-CRF Error Analysis ===")
print("Analysing boundary errors and entity type confusion...")

def get_errors(true_seqs, pred_seqs, raw_ds, n=5):
    errors = []
    for sent_idx, (ts, ps) in enumerate(zip(true_seqs, pred_seqs)):
        if ts != ps:
            sentence = raw_ds[sent_idx]["tokens"][:len(ts)]
            for tok_idx, (t, p) in enumerate(zip(ts, ps)):
                if t != p:
                    errors.append({
                        "sent_idx": sent_idx,
                        "tok_idx":  tok_idx,
                        "token":    sentence[tok_idx] if tok_idx < len(sentence) else "?",
                        "true":     t,
                        "pred":     p,
                        "context":  " ".join(sentence[max(0,tok_idx-3):tok_idx+4]),
                    })
    return errors[:n]

crf_errors  = get_errors(crf_true,       crf_pred,       test_ds, n=5)
bert_errors = get_errors(bert_true_seqs, bert_pred_seqs, test_ds, n=5)

print("\\nBiLSTM-CRF Errors (sample):")
for e in crf_errors:
    print(f"  Token: {e['token']!r:<15} True: {e['true']:<10} Pred: {e['pred']:<10} Context: {e['context']!r}")

print("\\nBERT NER Errors (sample):")
for e in bert_errors:
    print(f"  Token: {e['token']!r:<15} True: {e['true']:<10} Pred: {e['pred']:<10} Context: {e['context']!r}")

# ── Confusion heatmap for entity types ────────────────────────────────────────
entity_labels = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]

def confusion_from_seqs(true_seqs, pred_seqs, labels):
    n = len(labels); l2i = {l:i for i,l in enumerate(labels)}
    mat = np.zeros((n, n), dtype=int)
    for ts, ps in zip(true_seqs, pred_seqs):
        for t, p in zip(ts, ps):
            if t in l2i and p in l2i:
                mat[l2i[t], l2i[p]] += 1
    return mat

cm_bert = confusion_from_seqs(bert_true_seqs, bert_pred_seqs, entity_labels)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_bert, xticklabels=entity_labels, yticklabels=entity_labels,
            fmt="d", annot=True, cmap="Blues", ax=ax, annot_kws={"size": 7})
ax.set_title("BERT NER – Confusion Matrix (Test Set)")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout(); plt.savefig("ner_bert_confusion.png", dpi=150); plt.show()

print('''
Common Error Patterns
---------------------
1. Boundary errors  : Model predicts B-XXX where I-XXX expected (or vice versa).
                       E.g. New York City split as B-LOC B-LOC I-LOC instead of B-LOC I-LOC I-LOC.
2. Entity confusion : ORG vs LOC (Washington can be person, city, or organisation).
3. O -> B-XXX miss  : Short, rare entities (MISC) missed entirely.
4. B->I shift       : BiLSTM-CRF more prone to boundary shifts than BERT.
5. Unseen entities  : BiLSTM-CRF struggles with entities not in training vocabulary.
''')
"""

DISCUSSION = """\
## §6 – Discussion

### Role of Contextual Embeddings in NER

| Aspect | BiLSTM-CRF | BERT (cased) |
|---|---|---|
| Entity boundary detection | CRF enforces valid BIO transitions | Self-attention captures full context |
| Unseen words | Falls back to \\<UNK\\> embedding | Subword tokenisation covers any word |
| Capitalisation | Passed via vocabulary | Handled natively (cased model) |
| Pre-training | None | Wikipedia + BooksCorpus (3.3B words) |
| Approx F1 | ~85-87% | ~91-92% |

BERT's contextual embeddings capture long-range dependencies that LSTM misses.
Cased BERT preserves capitalisation — a strong signal for NER (e.g., *Paris* vs *paris*).
The CRF layer in BiLSTM-CRF enforces global tag consistency; BERT's linear head
does not, yet BERT still outperforms because its representations are richer.
"""

# ── Assemble ──────────────────────────────────────────────────────────────────
cells = [
    mc(TITLE,             "c00"),
    cc(INSTALL,           "c01"),
    cc(IMPORTS,           "c02"),
    mc(DATASET_HEADER,    "c03"),
    cc(LOAD_DATA,         "c04"),
    cc(DATA_STATS,        "c05"),
    mc(BILSTM_CRF_HEADER, "c06"),
    cc(BILSTM_CRF_DATA,   "c07"),
    cc(BILSTM_CRF_MODEL,  "c08"),
    cc(BILSTM_CRF_TRAIN,  "c09"),
    mc(BERT_NER_HEADER,   "c10"),
    cc(BERT_NER_DATA,     "c11"),
    cc(BERT_NER_TRAIN,    "c12"),
    cc(BERT_NER_EVAL,     "c13"),
    mc(COMPARISON_HEADER, "c14"),
    cc(COMPARISON,        "c15"),
    mc(ERROR_ANALYSIS_HEADER, "c16"),
    cc(ERROR_ANALYSIS,        "c16b"),
    mc(DISCUSSION,            "c17"),
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
