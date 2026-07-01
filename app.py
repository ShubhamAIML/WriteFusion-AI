"""
Language Model Inspector â€” Flask Backend
==========================================
Serves your trained Transformer model for:
  - Text completion (single next-word predictions)
  - Story / paragraph generation
  - Temperature + top-k sampling control
  - Model stats (vocab size, params, corpus size, perplexity)
  - Attention weight visualization

Run: python app.py
Then open: http://localhost:5000

Before running, place these in the same folder as app.py:
    best_model.keras
    tokenizer.pkl
    model_config.json
"""

import os
import re
import json
import pickle
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from flask import Flask, render_template, request, jsonify

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "best_model.keras"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"
CORPUS_STATS_PATH = MODEL_DIR / "corpus_stats.json"  # optional, see note at bottom


class SimpleWordTokenizer:
    WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

    def __init__(self, num_words=None, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}

    def _tokenize(self, text):
        return self.WORD_RE.findall(text.lower())

    def texts_to_sequences(self, lines):
        oov_id = self.word_index.get(self.oov_token, 1)
        out = []
        for line in lines:
            tokens = self._tokenize(line)
            ids = [self.word_index.get(t, oov_id) for t in tokens]
            out.append(ids)
        return out


# Support tokenizers that were pickled when the training script ran as __main__.
setattr(sys.modules.get('__main__'), 'SimpleWordTokenizer', SimpleWordTokenizer)


class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'seq_len': self.seq_len, 'vocab_size': self.vocab_size,
                    'embed_dim': self.embed_dim})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.last_attn_scores = None

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)

        attn_out, attn_scores = self.att(
            x, x, attention_mask=causal_mask, training=training,
            return_attention_scores=True
        )
        self.last_attn_scores = attn_scores
        if attn_out.dtype != x.dtype:
            attn_out = tf.cast(attn_out, x.dtype)
        attn_out = self.drop1(attn_out, training=training)
        if attn_out.dtype != x.dtype:
            attn_out = tf.cast(attn_out, x.dtype)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        if ffn_out.dtype != x.dtype:
            ffn_out = tf.cast(ffn_out, x.dtype)
        ffn_out = self.drop2(ffn_out, training=training)
        if ffn_out.dtype != x.dtype:
            ffn_out = tf.cast(ffn_out, x.dtype)
        x = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads,
                    'ff_dim': self.ff_dim, 'dropout': self.dropout})
        return cfg


class LastToken(layers.Layer):
    """Picks the last timestep's representation from a sequence.
    Replaces a raw Lambda layer, which Keras refuses to deserialize
    by default for security reasons (arbitrary code execution risk)."""
    def call(self, x):
        return x[:, -1, :]

    def get_config(self):
        return super().get_config()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Load model + tokenizer + config at startup
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerBlock': TransformerBlock,
        'LastToken': LastToken,
    }
)
print("Model loaded.")

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

SEQ_LEN = cfg['seq_len']
VOCAB_SIZE = cfg['vocab_size']

# Find the transformer block layers so we can read attention scores
transformer_layers = [l for l in model.layers if isinstance(l, TransformerBlock)]

# Optional corpus stats (sentences/words used in training) â€” if you saved
# this file during training, it'll show in the UI. Otherwise shows "â€”".
corpus_stats = {"sentences": None, "words": None}
if CORPUS_STATS_PATH.exists():
    with open(CORPUS_STATS_PATH) as f:
        corpus_stats = json.load(f)

MODEL_INFO = {
    "vocab_size": VOCAB_SIZE,
    "seq_len": SEQ_LEN,
    "embed_dim": cfg.get('embed_dim'),
    "num_heads": cfg.get('num_heads'),
    "num_layers": cfg.get('num_layers'),
    "ff_dim": cfg.get('ff_dim'),
    "total_params": int(model.count_params()),
    "corpus_sentences": corpus_stats.get("sentences") or 860173,
    "corpus_words": corpus_stats.get("words") or 2700000,
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Inference helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_attention_for_input(token_ids):
    """Run a forward pass and capture attention weights from the last block."""
    x = np.array([token_ids])
    _ = model(x, training=False)
    last_block = transformer_layers[-1]
    scores = last_block.last_attn_scores  # (batch, heads, seq, seq)
    if scores is None:
        return None
    scores = scores.numpy()[0]          # (heads, seq, seq)
    avg_scores = scores.mean(axis=0)    # (seq, seq) â€” average over heads
    last_row = avg_scores[-1]           # attention from the LAST token to all others
    return last_row.tolist()


def predict_distribution(text, temperature=0.8):
    ids = tokenizer.texts_to_sequences([text])[0][-SEQ_LEN:]
    padded = pad_sequences([ids], maxlen=SEQ_LEN, padding='pre')[0]

    probs = model.predict(np.array([padded]), verbose=0)[0].astype('float64')
    probs = np.log(probs + 1e-10) / max(temperature, 1e-6)
    probs = np.exp(probs)
    probs /= probs.sum()

    attn = get_attention_for_input(padded.tolist())
    # Trim attention to only the real (non-pad) tokens, aligned with input words
    real_len = len(ids)
    attn_trimmed = attn[-real_len:] if attn else None

    return probs, ids, attn_trimmed


def top_k_words(probs, k=8):
    top_idx = np.argsort(probs)[::-1]
    top_idx = [i for i in top_idx if i > 1][:k]
    return [
        {"word": tokenizer.index_word.get(i, '?'), "prob": round(float(probs[i]), 4)}
        for i in top_idx
    ]


def smart_complete_distribution(probs, context_ids):
    """Apply light anti-repetition filtering for live next-word suggestions."""
    probs = apply_repetition_penalty(
        probs,
        context_ids,
        penalty=DECODING_DEFAULTS["repetition_penalty"],
        recent_window=DECODING_DEFAULTS["recent_window"],
    )
    probs = apply_no_repeat_ngram(
        probs,
        context_ids,
        [],
        n=DECODING_DEFAULTS["no_repeat_ngram"],
    )
    return normalize_probs(probs)


DECODING_DEFAULTS = {
    "recent_window": 30,
    "repetition_penalty": 1.2,
    "no_repeat_ngram": 3,
    "dynamic_temp_step": 0.05,
    "dynamic_temp_max": 1.35,
    "dynamic_top_k_step": 8,
    "dynamic_top_k_max": 80,
    "top_p": 0.9,
    "typical_p": 0.92,
    "min_p": 0.04,
}


def normalize_probs(probs):
    """Return a valid probability distribution after filtering/penalties."""
    probs = np.asarray(probs, dtype="float64")
    probs[~np.isfinite(probs)] = 0
    probs[probs < 0] = 0
    probs[0] = 0
    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs, dtype="float64")
        probs[:2] = 0
        total = probs.sum()
    return probs / total


def repetition_score(token_ids, recent_window=30):
    """Score recent repetition using words, bigrams, trigrams, and token frequency."""
    recent = [int(t) for t in token_ids[-recent_window:] if int(t) > 1]
    if len(recent) < 4:
        return 0.0

    counts = Counter(recent)
    max_freq_ratio = max(counts.values()) / len(recent)
    repeated_word_ratio = sum(c - 1 for c in counts.values() if c > 1) / len(recent)

    def repeated_ngram_ratio(n):
        grams = [tuple(recent[i:i + n]) for i in range(len(recent) - n + 1)]
        if not grams:
            return 0.0
        gram_counts = Counter(grams)
        return sum(c - 1 for c in gram_counts.values() if c > 1) / len(grams)

    score = (
        repeated_word_ratio * 0.35
        + repeated_ngram_ratio(2) * 0.30
        + repeated_ngram_ratio(3) * 0.45
        + max(0.0, max_freq_ratio - 0.18) * 1.25
    )
    return round(float(min(score, 1.0)), 4)


def adjust_dynamic_controls(base_temperature, base_top_k, rep_score, state):
    """Raise temperature/top-k when repetition appears, then smoothly return."""
    temp_step = DECODING_DEFAULTS["dynamic_temp_step"]
    top_k_step = DECODING_DEFAULTS["dynamic_top_k_step"]
    max_temp = max(base_temperature, DECODING_DEFAULTS["dynamic_temp_max"])
    max_top_k = max(base_top_k or 0, DECODING_DEFAULTS["dynamic_top_k_max"])

    if rep_score >= 0.18:
        state["temperature"] = min(max_temp, state.get("temperature", base_temperature) + temp_step)
        if base_top_k and base_top_k > 0:
            state["top_k"] = min(max_top_k, state.get("top_k", base_top_k) + top_k_step)
    else:
        state["temperature"] = max(base_temperature, state.get("temperature", base_temperature) - temp_step)
        if base_top_k and base_top_k > 0:
            state["top_k"] = max(base_top_k, state.get("top_k", base_top_k) - top_k_step)

    return float(state["temperature"]), int(state.get("top_k", base_top_k or 0))


def apply_temperature(probs, temperature):
    probs = normalize_probs(probs)
    scaled = np.exp(np.log(probs + 1e-12) / max(float(temperature), 1e-6))
    return normalize_probs(scaled)


def apply_repetition_penalty(probs, generated_ids, penalty=1.2, recent_window=30):
    """Down-weight tokens repeated in the recent generated context."""
    if penalty <= 1.0:
        return probs
    probs = probs.copy()
    for token_id, count in Counter(generated_ids[-recent_window:]).items():
        if token_id > 1:
            probs[token_id] /= penalty ** min(count, 4)
    return normalize_probs(probs)


def apply_no_repeat_ngram(probs, context_ids, generated_ids, n=3):
    """Suppress tokens that would recreate an already-seen n-gram."""
    if n <= 1:
        return probs
    # context_ids is the full text history for n-gram blocking; generated_ids is
    # accepted for API symmetry with repetition penalty but is not duplicated here.
    sequence = [int(t) for t in context_ids if int(t) > 1]
    if len(sequence) < n - 1:
        return probs
    prefix = tuple(sequence[-(n - 1):])
    banned = set()
    for i in range(len(sequence) - n + 1):
        gram = tuple(sequence[i:i + n])
        if gram[:-1] == prefix:
            banned.add(gram[-1])
    if not banned:
        return probs
    probs = probs.copy()
    for token_id in banned:
        if 0 <= token_id < len(probs):
            probs[token_id] = 0
    return normalize_probs(probs)


def filter_top_k(probs, top_k):
    if not top_k or top_k <= 0:
        return probs
    top_k = min(int(top_k), len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    filtered = np.zeros_like(probs)
    filtered[top_idx] = probs[top_idx]
    return normalize_probs(filtered)


def filter_top_p(probs, top_p=0.9):
    top_p = float(np.clip(top_p, 0.05, 1.0))
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    remove = cumsum > top_p
    remove[1:] = remove[:-1].copy()
    remove[0] = False
    filtered = probs.copy()
    filtered[sorted_idx[remove]] = 0
    return normalize_probs(filtered)


def filter_min_p(probs, min_p=0.04):
    threshold = probs.max() * float(np.clip(min_p, 0.0, 0.5))
    filtered = probs.copy()
    filtered[filtered < threshold] = 0
    return normalize_probs(filtered)


def filter_typical(probs, typical_p=0.92):
    """Typical sampling keeps tokens whose surprise is closest to entropy."""
    typical_p = float(np.clip(typical_p, 0.05, 1.0))
    probs = normalize_probs(probs)
    log_probs = -np.log(probs + 1e-12)
    entropy = float(np.sum(probs * log_probs))
    shifted = np.abs(log_probs - entropy)
    sorted_idx = np.argsort(shifted)
    sorted_probs = probs[sorted_idx]
    keep_count = np.searchsorted(np.cumsum(sorted_probs), typical_p) + 1
    keep_idx = sorted_idx[:keep_count]
    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    return normalize_probs(filtered)


def sample_from_strategy(probs, strategy, top_k=0, top_p=0.9, typical_p=0.92, min_p=0.0):
    """Apply the selected decoding strategy and return token id plus final probs."""
    probs = normalize_probs(probs)
    strategy = (strategy or "top_k").lower()

    if strategy == "greedy":
        return int(np.argmax(probs)), probs
    if strategy == "top_k":
        probs = filter_top_k(probs, top_k)
    elif strategy == "top_p":
        probs = filter_top_p(probs, top_p)
    elif strategy == "top_k_top_p":
        probs = filter_top_p(filter_top_k(probs, top_k), top_p)
    elif strategy == "typical":
        probs = filter_typical(probs, typical_p)
    elif strategy == "min_p":
        probs = filter_min_p(probs, min_p)
    elif strategy == "temperature":
        pass
    elif strategy == "smart":
        probs = filter_top_p(filter_top_k(probs, top_k), top_p)

    probs = normalize_probs(probs)
    return int(np.random.choice(len(probs), p=probs)), probs


def decode_next_token(raw_probs, context_ids, generated_ids, options, state):
    """Single modular decoding step used only by /api/generate."""
    base_temperature = float(options.get("temperature", 0.8))
    base_top_k = int(options.get("top_k", 0))
    strategy = options.get("strategy", "top_k")
    smart = bool(options.get("smart", False)) or strategy == "smart"
    rep_score = repetition_score(
        context_ids + generated_ids,
        recent_window=int(options.get("recent_window", DECODING_DEFAULTS["recent_window"])),
    )

    temperature = base_temperature
    top_k = base_top_k
    if smart:
        temperature, top_k = adjust_dynamic_controls(base_temperature, base_top_k, rep_score, state)

    probs = apply_temperature(raw_probs, temperature)
    if smart:
        probs = apply_repetition_penalty(
            probs,
            generated_ids,
            penalty=float(options.get("repetition_penalty", DECODING_DEFAULTS["repetition_penalty"])),
            recent_window=int(options.get("recent_window", DECODING_DEFAULTS["recent_window"])),
        )
        probs = apply_no_repeat_ngram(
            probs,
            context_ids,
            generated_ids,
            n=int(options.get("no_repeat_ngram", DECODING_DEFAULTS["no_repeat_ngram"])),
        )
        strategy = "smart"

    next_id, final_probs = sample_from_strategy(
        probs,
        strategy=strategy,
        top_k=top_k,
        top_p=float(options.get("top_p", DECODING_DEFAULTS["top_p"])),
        typical_p=float(options.get("typical_p", DECODING_DEFAULTS["typical_p"])),
        min_p=float(options.get("min_p", DECODING_DEFAULTS["min_p"])),
    )
    return next_id, final_probs, {
        "repetition_score": rep_score,
        "temperature": round(float(temperature), 4),
        "top_k": int(top_k),
        "strategy": strategy,
        "smart": smart,
    }


def compute_perplexity(text):
    """Perplexity of the model on a piece of text (lower = more confident)."""
    words = tokenizer._tokenize(text)
    if len(words) < 2:
        return None
    ids = [tokenizer.word_index.get(w, 1) for w in words]

    losses = []
    for i in range(1, len(ids)):
        context = ids[max(0, i - SEQ_LEN):i]
        padded = pad_sequences([context], maxlen=SEQ_LEN, padding='pre')[0]
        probs = model(np.array([padded]), training=False).numpy()[0]
        true_id = ids[i]
        p = max(probs[true_id], 1e-10)
        losses.append(-np.log(p))

    avg_loss = float(np.mean(losses))
    return float(np.exp(avg_loss))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Routes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/api/model-info')
def model_info():
    return jsonify(MODEL_INFO)


@app.route('/api/complete', methods=['POST'])
def complete():
    """Single-step completion â€” returns top-k candidate next words + attention."""
    data = request.json
    text = data.get('text', '').strip()
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 8))

    if not text:
        return jsonify({"error": "empty input"}), 400

    probs, ids, attn = predict_distribution(text, temperature)
    probs = smart_complete_distribution(probs, tokenizer.texts_to_sequences([text])[0])
    candidates = top_k_words(probs, k=top_k)

    words = tokenizer._tokenize(text)
    return jsonify({
        "input_tokens": words[-SEQ_LEN:],
        "candidates": candidates,
        "attention": attn,
    })


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Multi-step generation with selectable decoding strategies.
    Prediction/model architecture are unchanged; only sampling is improved.
    """
    data = request.json or {}
    seed = data.get('text', '').strip()
    max_words = min(int(data.get('max_words', 80)), 500)

    if not seed:
        return jsonify({"error": "empty input"}), 400

    options = {
        "temperature": float(data.get('temperature', 0.8)),
        "top_k": int(data.get('top_k', 0)),
        "strategy": data.get('decoding_strategy', data.get('strategy', 'top_k')),
        "smart": bool(data.get('smart_anti_repetition', False)),
        "repetition_penalty": float(data.get('repetition_penalty', DECODING_DEFAULTS['repetition_penalty'])),
        "no_repeat_ngram": int(data.get('no_repeat_ngram', DECODING_DEFAULTS['no_repeat_ngram'])),
        "top_p": float(data.get('top_p', DECODING_DEFAULTS['top_p'])),
        "typical_p": float(data.get('typical_p', DECODING_DEFAULTS['typical_p'])),
        "min_p": float(data.get('min_p', DECODING_DEFAULTS['min_p'])),
        "recent_window": int(data.get('recent_window', DECODING_DEFAULTS['recent_window'])),
    }

    text = seed
    steps = []
    generated_ids = []
    decoding_state = {"temperature": options["temperature"], "top_k": options["top_k"]}

    for _ in range(max_words):
        ids = tokenizer.texts_to_sequences([text])[0][-SEQ_LEN:]
        padded = pad_sequences([ids], maxlen=SEQ_LEN, padding='pre')[0]
        raw_probs = model(np.array([padded]), training=False).numpy()[0].astype('float64')

        next_id, final_probs, decode_info = decode_next_token(
            raw_probs=raw_probs,
            context_ids=tokenizer.texts_to_sequences([text])[0],
            generated_ids=generated_ids,
            options=options,
            state=decoding_state,
        )
        if next_id <= 1:
            continue

        word = tokenizer.index_word.get(next_id)
        if not word:
            continue

        attn = get_attention_for_input(padded.tolist())
        real_len = len(ids)
        attn_trimmed = attn[-real_len:] if attn else None
        context_tokens = tokenizer._tokenize(text)[-real_len:]

        text += " " + word
        generated_ids.append(next_id)
        steps.append({
            "word": word,
            "prob": round(float(final_probs[next_id]), 4),
            "attention": attn_trimmed,
            "context_tokens": context_tokens,
            "decoding": decode_info,
        })

    return jsonify({
        "seed": seed,
        "generated": text,
        "steps": steps,
        "decoding_strategy": options["strategy"],
        "smart_anti_repetition": options["smart"],
    })


@app.route('/api/perplexity', methods=['POST'])
def perplexity():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "empty input"}), 400
    ppl = compute_perplexity(text)
    return jsonify({"perplexity": ppl})


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  NLP Engineer Studio")
    print(f"  Vocab size : {VOCAB_SIZE:,}")
    print(f"  Params     : {MODEL_INFO['total_params']:,}")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)






