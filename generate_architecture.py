"""
Generates the pipeline architecture diagram and saves it to docs/architecture.png.
Run once after cloning: python generate_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.makedirs("docs", exist_ok=True)

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

COLORS = {
    "data":      "#4A90D9",
    "preprocess":"#E67E22",
    "embed":     "#27AE60",
    "model":     "#8E44AD",
    "eval":      "#C0392B",
    "arrow":     "#555555",
    "header":    "#2C3E50",
}

def box(ax, x, y, w, h, label, sublabel="", color="#4A90D9", fontsize=10):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1", linewidth=1.5,
                          edgecolor="white", facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4, wrap=True)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.9, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["arrow"],
                                lw=1.8, mutation_scale=18), zorder=2)

def section_label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=9,
            color=COLORS["header"], fontstyle="italic", alpha=0.7)

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(10, 13.4, "Question & Answer Classification — Pipeline Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["header"])

# ── Row 1: Data Ingestion ──────────────────────────────────────────────────────
box(ax,  5, 12.2, 3.8, 0.85, "Training Set", "280K Q&A pairs  |  10 classes", COLORS["data"])
box(ax, 15, 12.2, 3.8, 0.85, "Test Set",     "60K Q&A pairs",                COLORS["data"])
section_label(ax, 10, 12.2, "─────────  Data Ingestion  ─────────")

# ── Row 2: Preprocessing ──────────────────────────────────────────────────────
steps = [
    ("Lowercase",   "Unicode normalise"),
    ("HTML/URL\nRemoval", "BeautifulSoup-style regex"),
    ("Emoji\nRemoval",    "ASCII encode/decode"),
    ("Punctuation\nRemoval", "string.punctuation"),
    ("Stop-word\nRemoval",   "NLTK English"),
    ("Lemmatisation", "WordNetLemmatizer"),
    ("Noise Filter", "Nonsense-ratio ≥ 0.5 → drop"),
]
y_pre = 10.6
xs = np.linspace(1.5, 18.5, len(steps))
for i, (lbl, sub) in enumerate(steps):
    box(ax, xs[i], y_pre, 2.35, 0.82, lbl, sub, COLORS["preprocess"], fontsize=8.5)
    if i > 0:
        arrow(ax, xs[i-1] + 1.18, y_pre, xs[i] - 1.18, y_pre)
section_label(ax, 10, 11.55, "─────────────────────────────  Text Preprocessing  ─────────────────────────────")

# ── Row 3: Feature Engineering ────────────────────────────────────────────────
y_feat = 9.1
feat_items = [
    (3.5,  "Bag-of-Words",    "CountVectorizer",         COLORS["embed"]),
    (7,    "TF-IDF",          "TfidfVectorizer",         COLORS["embed"]),
    (11,   "GloVe 6B",        "100-d  |  400K vocab",    COLORS["embed"]),
    (14.5, "Tokenisation +\nPad Sequences", "vocab=20K  |  maxlen=200", COLORS["embed"]),
    (18,   "Word2Vec\n(Skip-gram)", "dim=100  |  window=5",   COLORS["embed"]),
]
for x, lbl, sub, col in feat_items:
    box(ax, x, y_feat, 3.2, 0.82, lbl, sub, col, fontsize=8.5)

arrow(ax, 10, 10.18, 3.5, 9.52)
arrow(ax, 10, 10.18, 7,   9.52)
arrow(ax, 10, 10.18, 11,  9.52)
arrow(ax, 10, 10.18, 14.5,9.52)
arrow(ax, 10, 10.18, 18,  9.52)
section_label(ax, 10, 9.9, "─────────────────────  Feature Engineering / Word Embeddings  ──────────────────────")

# ── Row 4: Models ─────────────────────────────────────────────────────────────
y_mod = 7.45
models = [
    (1.7,  "Random\nForest",     COLORS["model"]),
    (4,    "Logistic\nRegression", COLORS["model"]),
    (6.3,  "Naive\nBayes",       COLORS["model"]),
    (8.6,  "DNN",                COLORS["model"]),
    (10.5, "SimpleRNN",          COLORS["model"]),
    (12.4, "LSTM",               COLORS["model"]),
    (14.3, "GRU",                COLORS["model"]),
    (16.2, "Bi-RNN",             COLORS["model"]),
    (17.9, "Bi-LSTM",            COLORS["model"]),
    (19.6, "Bi-GRU",             COLORS["model"]),
]
for x, lbl, col in models:
    box(ax, x, y_mod, 1.55, 0.8, lbl, "", col, fontsize=8)

# Arrows from feature rows to models
arrow(ax, 3.5, 8.7, 1.7, 7.85)
arrow(ax, 3.5, 8.7, 4,   7.85)
arrow(ax, 7,   8.7, 4,   7.85)
arrow(ax, 7,   8.7, 6.3, 7.85)
arrow(ax, 3.5, 8.7, 6.3, 7.85)
arrow(ax, 3.5, 8.7, 8.6, 7.85)
arrow(ax, 7,   8.7, 8.6, 7.85)
arrow(ax, 11,  8.7, 10.5,7.85)
arrow(ax, 11,  8.7, 12.4,7.85)
arrow(ax, 14.5,8.7, 12.4,7.85)
arrow(ax, 14.5,8.7, 14.3,7.85)
arrow(ax, 14.5,8.7, 16.2,7.85)
arrow(ax, 18,  8.7, 17.9,7.85)
arrow(ax, 18,  8.7, 19.6,7.85)
section_label(ax, 10, 8.2, "─────────────────────────────  Model Zoo  ─────────────────────────────────────────")

# ── Row 5: Training config ────────────────────────────────────────────────────
y_cfg = 6.05
cfg_items = [
    (3.5,  "Early Stopping",    "patience=3  |  monitor=val_loss"),
    (8.5,  "Adam Optimiser",    "sparse_cat_crossentropy loss"),
    (13.5, "Batch Size 128",    "max 30 epochs (RNN family)"),
    (18.5, "Dropout 0.2–0.3",   "regularisation"),
]
for x, lbl, sub in cfg_items:
    box(ax, x, y_cfg, 4.2, 0.75, lbl, sub, "#5D6D7E", fontsize=8.5)
section_label(ax, 10, 6.6, "─────────────────────────  Training Configuration  ──────────────────────────────────")

# ── Row 6: Evaluation ─────────────────────────────────────────────────────────
y_ev = 4.6
eval_items = [
    (3,   "Accuracy",            "Overall  &  per-class",  COLORS["eval"]),
    (7.3, "F1 Score",            "Macro  &  Weighted",     COLORS["eval"]),
    (12,  "Confusion Matrix",    "Count  &  Row-normalised", COLORS["eval"]),
    (16.8,"Classification\nReport", "Precision / Recall",  COLORS["eval"]),
]
for x, lbl, sub, col in eval_items:
    box(ax, x, y_ev, 3.6, 0.8, lbl, sub, col, fontsize=8.5)

for xi, _, _, _ in eval_items:
    arrow(ax, xi, 6.68, xi, 5.0)
section_label(ax, 10, 5.25, "────────────────────────────  Evaluation  ──────────────────────────────────────────")

# ── Row 7: Output artefacts ────────────────────────────────────────────────────
y_out = 3.2
out_items = [
    (3.5,  "images/output/",    "Confusion matrices\nTraining curves"),
    (8.5,  "images/input/",     "EDA plots\nWord cloud"),
    (13.5, "results/",          "metrics_summary.csv\nbaseline_models.csv  |  dl_models.csv"),
    (18,   "docs/",             "architecture.png"),
]
for x, lbl, sub in out_items:
    box(ax, x, y_out, 4.2, 0.85, lbl, sub, "#17A589", fontsize=8.5)

arrow(ax, 10, 4.2, 10, 3.65)
section_label(ax, 10, 3.9, "─────────────────────────  Saved Artefacts  ────────────────────────────────────────")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (COLORS["data"],      "Data"),
    (COLORS["preprocess"],"Preprocessing"),
    (COLORS["embed"],     "Feature Engineering"),
    (COLORS["model"],     "Models"),
    ("#5D6D7E",           "Training Config"),
    (COLORS["eval"],      "Evaluation"),
    ("#17A589",           "Artefacts"),
]
for i, (col, lbl) in enumerate(legend_items):
    px, py = 0.5 + i * 2.8, 2.0
    rect = FancyBboxPatch((px, py - 0.2), 0.4, 0.4, boxstyle="round,pad=0.05",
                          facecolor=col, edgecolor="white", lw=0.8, zorder=5)
    ax.add_patch(rect)
    ax.text(px + 0.55, py, lbl, va="center", fontsize=8, color=COLORS["header"])

ax.text(10, 1.35, "Dataset: Yahoo! Answers (sampled 3,000 train / 600 val / 500 test)  |  "
        "Best overall: DNN + TF-IDF (Acc 62.5%)  |  Best seq-embedding: Bi-GRU + GloVe (Acc 61.5%)",
        ha="center", va="center", fontsize=9, color="#555", style="italic")

plt.tight_layout(pad=0.5)
plt.savefig("docs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved: docs/architecture.png")
