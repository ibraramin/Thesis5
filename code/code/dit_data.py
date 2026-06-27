"""
dit_data.py
===========
Dataset, tokenization, and TF-IDF machinery for DITTP.

Responsibilities:
  * Load HuggingFaceH4/no_robots and turn conversations into (prompt, response) pairs.
  * Fit word-level TF-IDF over the response texts using the model's tokenizer
    as the analyzer (so the lexical units line up with the model's vocabulary).
  * Project the word-level TF-IDF down to sub-word tokens by averaging the
    scores of the words each sub-word could appear inside.
  * Precompute a single (V,) lookup tensor and cache it on disk.
  * Collate (prompt, response) pairs into the five-tensor batch expected by
    DITTPTrainer: input_ids, attention_mask, labels, position_type, weight_tensor.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _first_user_assistant(example: dict[str, Any]) -> dict[str, str] | None:
    """Pull the first user/assistant turn out of a no_robots conversation.

    no_robots conversations are always two turns (user, assistant). For SFT
    we only need the opening exchange; later turns would need a multi-turn
    collator we do not implement here.
    """
    messages = example.get("messages") or []
    user_msg = None
    assistant_msg = None
    for m in messages:
        if m.get("role") == "user" and user_msg is None:
            user_msg = m.get("content", "")
        elif m.get("role") == "assistant" and assistant_msg is None:
            assistant_msg = m.get("content", "")
        if user_msg is not None and assistant_msg is not None:
            break
    if not user_msg or not assistant_msg:
        return None
    return {"prompt": user_msg, "response": assistant_msg}


def load_no_robots(cache_dir: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return (train, test) lists of {'prompt', 'response'} dicts.

    Splits are taken straight from HuggingFaceH4/no_robots: 9.5k train and
    500 test examples. We drop any malformed rows defensively.
    """
    ds = load_dataset("HuggingFaceH4/no_robots", cache_dir=cache_dir)
    train_raw = ds.get("train", ds.get("train_sft"))
    test_raw = ds.get("test", ds.get("test_sft"))
    if train_raw is None or test_raw is None:
        raise RuntimeError("HuggingFaceH4/no_robots is missing expected train/test splits")

    def materialize(split):
        out = []
        for ex in split:
            pair = _first_user_assistant(ex)
            if pair is not None:
                out.append(pair)
        return out

    return materialize(train_raw), materialize(test_raw)


# ---------------------------------------------------------------------------
# TF-IDF: word level, fit on response column
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _tokenizer_analyzer(tokenizer) -> Callable[[str], list[str]]:
    """Build a TfidfVectorizer analyzer that mirrors the HF tokenizer's
    pre-tokenization. We deliberately stop at the word boundary rather than
    at sub-words; the whole point of projecting down to sub-words below is
    to bridge the gap between corpus statistics and BPE fragments.
    """

    def analyzer(text: str) -> list[str]:
        # The HF tokenizer's pre-tokenizer already splits on the same
        # boundaries as our regex for the languages we care about. Using
        # tokenize(..., add_special_tokens=False) gives us the cleanest
        # answer without dragging the BPE merge step in.
        if not text:
            return []
        try:
            words = tokenizer.tokenize(text, add_special_tokens=False)
            return [w for w in words if w.strip()]
        except Exception:
            return _WORD_RE.findall(text.lower())

    return analyzer


def fit_word_tfidf(
    train_split: list[dict],
    tokenizer,
    min_df: int = 2,
    max_df: float = 0.95,
    direction: str = "standard",
) -> dict[str, float]:
    """Fit a TF-IDF vectorizer over the response column.

    The returned dict maps word -> score in the [0.5, 2.0] range. The
    bounds are mandatory: 0.5 keeps frequent tokens from being zeroed out
    entirely, and 2.0 caps the gradient magnitude on rare tokens so a
    handful of long-tail words cannot dominate the loss.

    direction="standard" maps the LOWEST tfidf score to 0.5 and the
    HIGHEST to 2.0 (rare tokens weighted more -- the original DITTP
    hypothesis). direction="reverse" inverts that mapping within the
    same [0.5, 2.0] envelope so common tokens get weight 2.0 and rare
    tokens get weight 0.5. The clip range is preserved exactly: this
    is a within-envelope inversion, not a scale change.
    """
    corpus = [ex["response"] for ex in train_split if ex.get("response")]
    vectorizer = TfidfVectorizer(
        analyzer=_tokenizer_analyzer(tokenizer),
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        token_pattern=None,  # we provide our own analyzer
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = matrix.max(axis=0).toarray().ravel()  # max tfidf per term across docs
    terms = vectorizer.get_feature_names_out()

    # Min-max normalize to [0.5, 2.0] with safety against the degenerate
    # case where the entire response column is a single word.
    s_min = float(scores.min())
    s_max = float(scores.max())
    if s_max <= s_min:
        normalized = [1.0] * len(scores)
    else:
        span = s_max - s_min
        normalized = [0.5 + 1.5 * (float(s) - s_min) / span for s in scores]

    # Reverse the mapping: lowest score -> 2.0, highest -> 0.5. This is a
    # symmetric flip around 1.25 (the center of [0.5, 2.0]), so each
    # weight w becomes 2.5 - w.
    if direction == "reverse":
        normalized = [2.5 - w for w in normalized]
    elif direction != "standard":
        raise ValueError(
            f"Unknown tfidf direction {direction!r}; expected 'standard' or 'reverse'"
        )

    return {term: float(w) for term, w in zip(terms, normalized)}


# ---------------------------------------------------------------------------
# TF-IDF: project to sub-word tokens
# ---------------------------------------------------------------------------

_WORD_SPLIT_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def project_word_tfidf_to_subwords(
    word_tfidf: dict[str, float],
    tokenizer,
) -> dict[int, float]:
    """For every sub-word in the vocab, average the word scores of the words
    it could be part of. Sub-words that do not cleanly line up with any
    single word fall back to the corpus median (or 1.0 if undefined).
    """
    vocab = tokenizer.get_vocab()
    # Pre-compute the median so we can hand a stable default to sub-words
    # whose parent words all live outside the fitted vocabulary.
    if word_tfidf:
        sorted_scores = sorted(word_tfidf.values())
        median = sorted_scores[len(sorted_scores) // 2]
    else:
        median = 1.0

    out: dict[int, float] = {}
    for token_str, token_id in vocab.items():
        # Most BPE tokens start with a leading space; strip it so that
        # e.g. " the" and "the" line up with the same word score.
        clean = token_str.lstrip("Ġ▁").lower()
        if not clean:
            out[token_id] = median
            continue
        parent_words = _WORD_SPLIT_RE.findall(clean)
        if not parent_words:
            out[token_id] = median
            continue
        scores = [word_tfidf[w] for w in parent_words if w in word_tfidf]
        if not scores:
            out[token_id] = median
        else:
            out[token_id] = sum(scores) / len(scores)

    return out


# ---------------------------------------------------------------------------
# TF-IDF: cached tensor
# ---------------------------------------------------------------------------

def _cache_path(
    model_name: str,
    cache_dir: str = "./cache",
    direction: str = "standard",
) -> str:
    safe = model_name.replace("/", "_").replace("\\", "_")
    os.makedirs(cache_dir, exist_ok=True)
    # Direction must be in the cache key: a "reverse" run would otherwise
    # load the standard-direction cache and silently use the wrong weights.
    suffix = "" if direction == "standard" else f"_{direction}"
    return os.path.join(cache_dir, f"tfidf_{safe}{suffix}.pt")


def build_tfidf_tensor(
    tokenizer,
    train_split: list[dict],
    model_name: str,
    cache_dir: str = "./cache",
    force: bool = False,
    direction: str = "standard",
) -> torch.Tensor:
    """Fit word TF-IDF, project to sub-words, and return a (V,) float tensor.

    The result is cached as a .pt file keyed on (model_name, direction)
    so the expensive sklearn fit only happens once per (tokenizer, direction)
    pair. Standard-direction runs keep the legacy filename so existing
    caches built before the direction key was added still load correctly.
    """
    path = _cache_path(model_name, cache_dir=cache_dir, direction=direction)
    if os.path.exists(path) and not force:
        tensor = torch.load(path, map_location="cpu")
        if tensor.shape[0] == tokenizer.vocab_size:
            return tensor.float()

    word_tfidf = fit_word_tfidf(train_split, tokenizer, direction=direction)
    subword_tfidf = project_word_tfidf_to_subwords(word_tfidf, tokenizer)
    vocab_size = tokenizer.vocab_size
    tensor = torch.ones(vocab_size, dtype=torch.float32)
    for tok_id, score in subword_tfidf.items():
        if 0 <= tok_id < vocab_size:
            tensor[tok_id] = float(score)
    torch.save(tensor, path)
    return tensor


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class DITCollator:
    """Build training batches with the five tensors DITTPTrainer expects.

    For each (prompt, response) pair we:
      1. Apply the tokenizer's chat template to the full conversation
         (prompt + response) and tokenize once.
      2. Re-tokenize the prompt-only template to find the boundary index
         between the instruction and the response.
      3. Mark the prompt span as position_type=1, the response span as
         position_type=2, and any padding as position_type=0.
      4. Set labels=-100 over padding always. In standard SFT (im_mode=False)
         also mask the prompt tokens; in Instruction Modelling mode
         (im_mode=True) the prompt tokens carry their real ids so the
         loss is computed over them.
      5. Set per-token weights: prompt_weight for prompt (or 1.0 in IM mode,
         which the Trainer subclass enforces), TF-IDF lookup (or 1.0) for
         response, 0 for padding.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        prompt_weight: float = 0.0,
        tfidf_tensor: torch.Tensor | None = None,
        im_mode: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.prompt_weight = float(prompt_weight)
        self.tfidf_tensor = tfidf_tensor
        # Instruction Modelling (Shi et al. NeurIPS 2024): include the
        # prompt tokens in the loss with their real ids, instead of the
        # standard SFT pattern of masking them with -100. The Trainer
        # subclass is responsible for forcing the prompt weight to 1.0
        # when this flag is set; here we just stop masking the labels.
        self.im_mode = bool(im_mode)

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Tokenize just the prompt with the chat template's generation prefix.

        The length of this is the boundary index inside the full sequence.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Tokenizers without a chat template (rare) fall back to a
            # plain "Instruction: ...\n" prompt so the pipeline still runs.
            prompt_text = f"Instruction: {prompt}\nResponse:"
        ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        return ids

    def _encode_full(self, prompt: str, response: str) -> tuple[list[int], int]:
        """Tokenize the full conversation and return (ids, prompt_token_count)."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            full_text = f"Instruction: {prompt}\nResponse: {response}"
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        prompt_ids = self._encode_prompt(prompt)
        # Clamp the prompt length to the actual full length (truncation may
        # have cut the response short, in which case everything after the
        # truncated prompt is still "response" as far as the model is told).
        prompt_len = min(len(prompt_ids), len(full_ids))
        return full_ids, prompt_len

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        # Truncate the batch defensively. The Trainer pads dynamically.
        max_len = self.max_length
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        all_ids: list[list[int]] = []
        prompt_lens: list[int] = []
        for ex in batch:
            ids, p_len = self._encode_full(ex["prompt"], ex["response"])
            all_ids.append(ids)
            prompt_lens.append(p_len)
        max_len = min(max_len, max(len(x) for x in all_ids))

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels_list: list[list[int]] = []
        position_types: list[list[int]] = []
        weights_list: list[list[float]] = []

        tfidf = self.tfidf_tensor
        for ids, p_len in zip(all_ids, prompt_lens):
            # Truncate but keep the original length so we know where real
            # tokens end and padding begins.
            ids = ids[:max_len]
            p_len = min(p_len, len(ids))
            actual_len = len(ids)
            attn = [1] * actual_len + [0] * (max_len - actual_len)
            ids = ids + [pad_id] * (max_len - actual_len)

            # labels: -100 over padding always. The prompt span is masked
            # in standard SFT (mask=-100 for i < p_len) but revealed in
            # Instruction Modelling mode (Shi et al. NeurIPS 2024), where
            # the loss is computed over the prompt tokens as well.
            labels = [-100] * max_len
            prompt_start = 0 if self.im_mode else p_len
            for i in range(prompt_start, actual_len):
                labels[i] = ids[i]

            # position_type: 1=prompt, 2=response, 0=padding/masked.
            pos_types = (
                [1] * p_len
                + [2] * (actual_len - p_len)
                + [0] * (max_len - actual_len)
            )

            # weights: prompt_weight for prompt, TF-IDF (or 1.0) for
            # response, 0 for padding.
            weights = [0.0] * max_len
            for i in range(p_len):
                weights[i] = self.prompt_weight
            for i in range(p_len, actual_len):
                tok = ids[i]
                if tfidf is not None and 0 <= tok < tfidf.shape[0]:
                    weights[i] = float(tfidf[tok].item())
                else:
                    weights[i] = 1.0

            input_ids.append(ids)
            attention_masks.append(attn)
            labels_list.append(labels)
            position_types.append(pos_types)
            weights_list.append(weights)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "position_type": torch.tensor(position_types, dtype=torch.long),
            "weight_tensor": torch.tensor(weights_list, dtype=torch.float32),
        }
