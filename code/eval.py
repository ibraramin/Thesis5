"""
eval.py
=======
Seven deterministic evaluation metrics for DITTP-trained adapters.

The functions all take a (model, tokenizer) pair. The caller is responsible
for merging the LoRA adapter into the base model beforehand; the eval
functions themselves do not touch PEFT. This keeps the eval math focused
on the metrics rather than on adapter plumbing.

Metrics:
  1. eval_perplexity            -- standard PPL over the no_robots test split
  2. eval_verbatim_memorization -- longest common substring vs. training refs
                                   (also feeds the diversity metric)
  3. eval_arc_challenge         -- exact-match A/B/C/D on ai2_arc
  4. eval_format_robustness     -- variance of next-token probs across perturbations
  5. _wiki_ppl_eval             -- cross-domain PPL on wikitext-103 raw v1 test
  6. _hellaswag_eval            -- 4-way multiple choice accuracy on HellaSwag val
  7. _diversity_eval            -- distinct-1/2 + repetition-4 over the LCS outputs
"""

from __future__ import annotations

import collections
import difflib
import json
import math
import random
import re
import string
from typing import Any

import torch
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate_left(tokenizer, input_ids: torch.Tensor, max_len: int) -> torch.Tensor:
    """Right-pad-free left truncation. The most recent tokens always survive."""
    if input_ids.size(1) <= max_len:
        return input_ids
    return input_ids[:, -max_len:]


def _build_chat_prompt(tokenizer, user_text: str) -> str:
    """Apply the tokenizer's chat template, fall back to a plain prompt."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"Instruction: {user_text}\nResponse:"


def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
) -> str:
    """Greedy (or sampled) generation helper. Returns the decoded continuation."""
    prompt_text = _build_chat_prompt(tokenizer, prompt)
    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    do_sample: bool = False,
) -> list[str]:
    """Batched generation with left-padding. Returns decoded continuations, one per prompt.

    With tokenizer.padding_side = "left", all prompts in the batch are padded to the
    longest prompt, so input_ids has shape (batch, max_prompt_len). generate() with
    attention_mask produces output of shape (batch, max_prompt_len + new_len). We
    slice out[:, input_len:] to recover only the newly generated tokens.

    With do_sample=False (greedy), per-example results are identical to single-example
    _generate() calls in the same order (up to bf16 reduction order on different
    hardware; on the same hardware with the same input shape the argmax is stable).
    """
    chat_prompts = [_build_chat_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    new_tokens = out[:, input_len:]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

def _compute_ppl_from_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int = 1024,
) -> tuple[float, int]:
    """Core batched perplexity computation over a list of raw texts.

    Shared by the no_robots and wikitext perplexity metrics. Tokenizes the
    texts in batches (with dynamic padding and right-side truncation), runs
    a forward pass per batch, and accumulates the per-token NLL across
    batches. Returns (perplexity, n_tokens). The caller is responsible for
    model.eval() and for any pre-tokenization/filtering of the texts.
    """
    nlls: list[float] = []
    n_tokens = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal nlls, n_tokens
        if not batch:
            return
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = model(**enc, labels=labels)
        # out.loss is mean CE across non-masked tokens; recover total NLL.
        active = (labels != -100).sum().item()
        nll = float(out.loss.item()) * active
        nlls.append(nll)
        n_tokens += active
        batch.clear()

    for t in texts:
        batch.append(t)
        if len(batch) >= batch_size:
            flush()
    flush()

    avg_nll = sum(nlls) / max(n_tokens, 1)
    return float(torch.tensor(avg_nll).exp().item()), int(n_tokens)


def eval_perplexity(
    model,
    tokenizer,
    test_data: list[dict],
    batch_size: int = 16,
    max_length: int = 1024,
) -> dict[str, float]:
    """Standard sliding-window-free perplexity over the no_robots test split.

    Each example is the full prompt+response conversation tokenized once.
    We use the model's built-in loss with label = input_ids, which is the
    conventional causal-LM PPL formulation. Delegates the batched math to
    _compute_ppl_from_texts so wikitext can reuse the same logic.
    """
    model.eval()
    texts = [_build_chat_prompt(tokenizer, ex["prompt"]) + ex["response"] for ex in test_data]
    ppl, n_tokens = _compute_ppl_from_texts(
        model, tokenizer, texts, batch_size=batch_size, max_length=max_length
    )
    return {"perplexity": ppl, "n_tokens": n_tokens}


# ---------------------------------------------------------------------------
# 2. Verbatim memorization
# ---------------------------------------------------------------------------

def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by a and b.

    difflib.SequenceMatcher gives the same answer but lazily, in O(N*M).
    For our 100-example x 128-token scale that's fine and the implementation
    is one line.
    """
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    match = sm.find_longest_match(0, len(a), 0, len(b))
    return int(match.size)


def eval_verbatim_memorization(
    model,
    tokenizer,
    train_data: list[dict],
    n: int = 100,
    max_new_tokens: int = 128,
    seed: int = 42,
    batch_size: int = 4,
    generations_sink: list[str] | None = None,
) -> dict[str, float]:
    """Sample n training prompts, generate, and measure longest common
    substring with the gold response. We report mean and std across n.

    The inner loop batches generation calls (left-padded). batch_size=4
    is the conservative default for max_new_tokens=128; the CLI default
    is 8 (overrideable via --gen-batch-size) and the user can drop it
    back to 4 if the 3B model OOMs at batch=8 * 128 new tokens.

    If `generations_sink` is provided, each decoded continuation is
    appended to it in order. This is how the diversity metric reuses
    these generations without re-running inference. When None, the
    generations are discarded (the default; behavior is unchanged from
    the version that did not support diversity).
    """
    model.eval()
    rng = random.Random(seed)
    sample = rng.sample(train_data, k=min(n, len(train_data)))

    lengths: list[int] = []
    for start in range(0, len(sample), batch_size):
        batch_ex = sample[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch_ex]
        gens = _generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        if generations_sink is not None:
            generations_sink.extend(gens)
        for ex, gen in zip(batch_ex, gens):
            lcs = _longest_common_substring(gen, ex["response"])
            lengths.append(lcs)

    if not lengths:
        return {"lcs_mean": 0.0, "lcs_std": 0.0, "n": 0}
    t = torch.tensor(lengths, dtype=torch.float32)
    return {
        "lcs_mean": float(t.mean().item()),
        "lcs_std": float(t.std(unbiased=False).item()),
        "n": int(len(lengths)),
    }


# ---------------------------------------------------------------------------
# 3. ARC-Challenge
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_letter(text: str) -> str | None:
    """Pull the first standalone A/B/C/D out of a generation."""
    m = _LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()
    # Fall back to "the answer is X" style phrasing.
    lowered = text.lower()
    for letter in "abcd":
        if f"answer is {letter}" in lowered or f"answer: {letter}" in lowered:
            return letter.upper()
    return None


def eval_arc_challenge(
    model,
    tokenizer,
    n: int = 1172,
    # ARC answers are single letters; 4 covers the letter + trailing EOS
    max_new_tokens: int = 4,
    cache_dir: str | None = None,
    batch_size: int = 8,
) -> dict[str, float]:
    """Exact-match accuracy on allenai/ai2_arc (ARC-Challenge test split).

    The model sees the question + the four options, generates a short
    continuation, and we extract the first A/B/C/D letter.

    The outer loop batches examples in groups of `batch_size` and
    generates them together via _generate_batch (left-padded). The
    per-example letter extraction and counter logic is preserved
    exactly, so the only difference vs. the per-example loop is one
    shared forward pass per batch.
    """
    model.eval()
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=cache_dir)
    test = ds["test"]
    n = min(n, len(test))
    correct = 0
    answered = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_exs = [test[i] for i in range(start, end)]
        prompts: list[str] = []
        golds: list[str] = []
        for ex in batch_exs:
            question = ex["question"]
            labels = ex["choices"]["label"]
            texts = ex["choices"]["text"]
            options = "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(labels, texts))
            prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with a single letter (A, B, C, or D)."
            prompts.append(prompt)
            golds.append(ex["answerKey"].strip().upper())
        gens = _generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
        for gen, gold in zip(gens, golds):
            pred = _extract_letter(gen)
            if pred is not None:
                answered += 1
                if pred == gold:
                    correct += 1
    return {
        "arc_accuracy": correct / max(n, 1),
        "arc_answered": answered / max(n, 1),
        "arc_n": int(n),
    }


# ---------------------------------------------------------------------------
# 4. Format robustness
# ---------------------------------------------------------------------------

_PERTURBATIONS: list[str] = [
    "capitalization",
    "synonym",
    "whitespace",
    "paraphrase",
    "reorder",
]


_SYNONYMS = {
    "what": "which",
    "which": "what",
    "how": "in what way",
    "why": "for what reason",
    "when": "at what time",
    "can": "could",
    "could": "can",
    "should": "ought to",
    "the": "a",
    "a": "the",
    "is": "appears",
    "are": "appear",
}


def _perturb(text: str, kind: str, rng: random.Random) -> str:
    """Apply one of five deterministic-ish perturbations to a prompt."""
    if kind == "capitalization":
        # Randomly flip 30% of alphabetic chars.
        out = []
        for ch in text:
            if ch.isalpha() and rng.random() < 0.3:
                out.append(ch.upper() if ch.islower() else ch.lower())
            else:
                out.append(ch)
        return "".join(out)
    if kind == "synonym":
        words: list[str] = re.findall(r"\w+|\W+", text)
        pieces: list[str] = [
            _SYNONYMS.get(w.lower(), w) if w.isalpha() else w
            for w in words
        ]
        return "".join(pieces)
    if kind == "whitespace":
        # Inject extra spaces and strip the trailing newline.
        return re.sub(r"\s+", "  ", text).strip()
    if kind == "paraphrase":
        # Light paraphrase: prepend a hedge and append a question mark if missing.
        prefix = rng.choice(["Please answer: ", "I'd like to know: ", "Quick question - "])
        suffix = "" if text.rstrip().endswith("?") else " Thanks."
        return prefix + text.rstrip() + suffix
    if kind == "reorder":
        # Shuffle the order of word groups separated by commas.
        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 2:
            return text
        head, *rest = parts
        rng.shuffle(rest)
        return head + ", " + ", ".join(rest)
    return text


def _next_token_probs(model, tokenizer, prompt: str) -> torch.Tensor:
    """Forward pass returning the softmax distribution at the final position."""
    text = _build_chat_prompt(tokenizer, prompt)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model(**enc)
    last_logits = out.logits[0, -1, :].float()
    return torch.softmax(last_logits, dim=-1)


def _next_token_probs_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int = 1024,
) -> torch.Tensor:
    """Batched forward pass returning softmax distributions at the final position.

    Returns shape (batch, vocab_size). Uses left-padding so the final
    column aligns across all rows in the batch -- each row's last
    position is the last token of that row's actual prompt.
    """
    chat_prompts = [_build_chat_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(
        chat_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        out = model(**enc)
    last_logits = out.logits[:, -1, :].float()
    return torch.softmax(last_logits, dim=-1)


def eval_format_robustness(
    model,
    tokenizer,
    n: int = 100,
    n_perturbations: int = 5,
    max_new_tokens: int = 64,
    seed: int = 42,
    cache_dir: str | None = None,
    batch_size_questions: int = 8,
) -> dict[str, float]:
    """Mean variance of next-token probabilities across prompt perturbations.

    We pull n short questions from no_robots (or fall back to ARC if no_robots
    isn't available), perturb each one five times, then measure the average
    total-variation distance of the next-token probability vectors.

    The inner loop batches `batch_size_questions` questions at a time. For
    each batch we build `batch_size_questions * n_perturbations` perturbed
    prompts, run a single forward pass, and compute the per-question
    variance across the perturbation axis. Variance is order-invariant, so
    the aggregated mean is identical to the per-question loop.
    """
    model.eval()
    rng = random.Random(seed)

    try:
        ds = load_dataset("HuggingFaceH4/no_robots", cache_dir=cache_dir, split="test")
        questions = []
        for ex in ds:
            for m in ex.get("messages", []):
                if m.get("role") == "user":
                    questions.append(m["content"])
                    break
    except Exception:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", cache_dir=cache_dir, split="test")
        questions = [ex["question"] for ex in ds]

    rng.shuffle(questions)
    questions = questions[:n]

    perturbations = _PERTURBATIONS[:n_perturbations]
    variances: list[float] = []
    for start in range(0, len(questions), batch_size_questions):
        batch_q = questions[start:start + batch_size_questions]
        all_prompts = [_perturb(q, p, rng) for q in batch_q for p in perturbations]
        all_probs = _next_token_probs_batch(model, tokenizer, all_prompts)
        # (B*5, V) -> (B, 5, V)
        B = len(batch_q)
        P = len(perturbations)
        V = all_probs.shape[-1]
        reshaped = all_probs.view(B, P, V)
        # Variance across the 5 perturbations for each (q, v) pair.
        per_token_var = reshaped.var(dim=1, unbiased=False)
        # Mean over vocab dim per question.
        variances.extend(per_token_var.mean(dim=-1).tolist())

    if not variances:
        return {"format_variance_mean": 0.0, "format_variance_std": 0.0, "n": 0}
    t = torch.tensor(variances, dtype=torch.float32)
    return {
        "format_variance_mean": float(t.mean().item()),
        "format_variance_std": float(t.std(unbiased=False).item()),
        "n": int(len(variances)),
    }


# ---------------------------------------------------------------------------
# 5. Wikitext-103 cross-domain perplexity
# ---------------------------------------------------------------------------

def _wiki_ppl_eval(
    model,
    tokenizer,
    batch_size: int = 16,
    max_examples: int = 0,
    max_length: int = 1024,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Cross-domain perplexity on wikitext-103-raw-v1 test split.

    Filters out empty/short lines (len < 50 chars), then runs the same
    batched PPL computation as eval_perplexity. max_examples=0 means
    use the full filtered test split. Cross-domain PPL catches the case
    where a model overfits to the in-domain no_robots distribution and
    loses general language modeling quality.
    """
    model.eval()
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="test", cache_dir=cache_dir
    )
    texts = [t for t in ds["text"] if len(t.strip()) >= 50]
    if max_examples > 0:
        texts = texts[:max_examples]
    if not texts:
        return {"perplexity": float("nan"), "n_tokens": 0, "n_filtered_docs": 0}
    print(f"[eval] wiki_ppl: {len(texts)} wikitext-103 docs after length filter")
    ppl, n_tokens = _compute_ppl_from_texts(
        model, tokenizer, texts, batch_size=batch_size, max_length=max_length
    )
    return {
        "perplexity": ppl,
        "n_tokens": n_tokens,
        "n_filtered_docs": len(texts),
    }


# ---------------------------------------------------------------------------
# 6. HellaSwag 4-way multiple choice
# ---------------------------------------------------------------------------

def _score_ending_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ctx_len: int,
    total_len: int,
) -> float:
    """Sum the log-prob of the ending tokens under the model.

    The full text (ctx + ending) is in input_ids. We use the logits at
    positions [ctx_len-1, total_len-1) to predict the target tokens at
    positions [ctx_len, total_len). Sum of per-token log-probs is the
    standard HellaSwag "log-likelihood" score the user requested.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
    # Edge case: ctx_len == total_len means the ending is empty; assign
    # the worst possible score so this option is never picked.
    if ctx_len >= total_len:
        return float("-inf")
    pred = log_probs[ctx_len - 1 : total_len - 1, :]
    targets = input_ids[0, ctx_len:total_len]
    token_log_probs = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def _hellaswag_eval(
    model,
    tokenizer,
    n: int = 1000,
    batch_size: int = 8,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """4-way multiple choice on Rowan/hellaswag (validation split).

    Standard HellaSwag eval: for each example, score each of the 4
    (ctx + ending) continuations by sum log-prob of the ending tokens,
    pick the option with the highest score, and compare to the gold
    label. No chat template -- the dataset is not a chat. Accuracy and
    the standard-error-of-the-mean (sqrt(p(1-p)/n)) are returned.
    """
    model.eval()
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", cache_dir=cache_dir)
    except Exception as e:
        return {"error": f"failed to load HellaSwag: {e}", "accuracy": 0.0, "se": 0.0, "n": 0}

    n = min(n, len(ds))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    correct = 0

    for batch_start in range(0, n, batch_size):
        batch_exs = [ds[i] for i in range(batch_start, min(batch_start + batch_size, n))]
        # Tokenize all 4*batch_size (ctx + ending) texts; left-pad to a
        # shared length so the single forward pass is valid.
        all_ids: list[list[int]] = []
        ctx_lens: list[int] = []
        gold_labels: list[int] = []
        for ex in batch_exs:
            ctx_text = ex["ctx"]
            ctx_ids = tokenizer(ctx_text, add_special_tokens=True)["input_ids"]
            try:
                gold_labels.append(int(ex["label"]))
            except (TypeError, ValueError):
                gold_labels.append(-1)
            for ending in ex["endings"]:
                full_text = ctx_text + " " + ending
                full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
                all_ids.append(full_ids)
                ctx_lens.append(len(ctx_ids))

        max_len = max(len(ids) for ids in all_ids)
        padded_ids: list[list[int]] = []
        padded_mask: list[list[int]] = []
        for ids in all_ids:
            n_pad = max_len - len(ids)
            padded_ids.append([pad_id] * n_pad + ids)
            padded_mask.append([0] * n_pad + [1] * len(ids))
        input_ids = torch.tensor(padded_ids, dtype=torch.long).to(model.device)
        attention_mask = torch.tensor(padded_mask, dtype=torch.long).to(model.device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Score each (example, option) pair, then argmax over the 4 options.
        for j, ex in enumerate(batch_exs):
            scores: list[float] = []
            for k in range(4):
                opt_idx = j * 4 + k
                ctx_len = ctx_lens[opt_idx]
                total_len = len(all_ids[opt_idx])
                if ctx_len >= total_len:
                    scores.append(float("-inf"))
                    continue
                pred = log_probs[opt_idx, ctx_len - 1 : total_len - 1, :]
                targets = input_ids[opt_idx, ctx_len:total_len]
                token_log_probs = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_log_probs.sum().item()))
            pred_idx = max(range(4), key=lambda i: scores[i])
            if pred_idx == gold_labels[j]:
                correct += 1

    accuracy = correct / max(n, 1)
    # Standard error of the mean for a Bernoulli: sqrt(p(1-p)/n).
    se = math.sqrt(accuracy * (1.0 - accuracy) / max(n, 1))
    return {
        "accuracy": accuracy,
        "se": se,
        "n": int(n),
        "correct": int(correct),
    }


# ---------------------------------------------------------------------------
# 7. Diversity (distinct-1, distinct-2, repetition-4)
# ---------------------------------------------------------------------------

_WORD_SPLIT_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize_for_diversity(text: str) -> list[str]:
    """Simple word-level tokenization for diversity metrics.

    Uses a regex word split (lowercased) rather than the model's BPE
    tokenizer. Diversity metrics are about surface-level lexical variety
    and repetition, where BPE fragmentation would distort the numbers:
    a single common word split into 4 BPE pieces would look like 4
    distinct unigrams and inflate distinct-1.
    """
    return _WORD_SPLIT_RE.findall(text.lower())


def _diversity_eval(generations: list[str]) -> dict[str, Any]:
    """Compute distinct-1, distinct-2, and repetition-4 over a list of texts.

    Concatenates all generations with a single space, tokenizes as
    whitespace-separated lowercased words, then:
      * distinct_1 = unique unigrams / total unigrams
      * distinct_2 = unique bigrams / total bigrams
      * repetition_4 = fraction of unique 4-grams whose count > 1
                       (i.e. how much of the 4-gram vocabulary is repeated
                       at least once -- a low value means high diversity)
    """
    if not generations:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_4": 0.0, "n": 0}
    tokens: list[str] = []
    for g in generations:
        tokens.extend(_tokenize_for_diversity(g))
    if not tokens:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_4": 0.0, "n": len(generations)}

    unigrams = tokens
    bigrams = list(zip(tokens, tokens[1:]))
    fourgrams = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))

    distinct_1 = len(set(unigrams)) / max(len(unigrams), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)

    if fourgrams:
        counts = collections.Counter(fourgrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        repetition_4 = repeated / len(counts)
    else:
        repetition_4 = 0.0

    return {
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "repetition_4": float(repetition_4),
        "n": int(len(generations)),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all four metrics against a merged (base + adapter) model.

    Usage:
        python eval.py --adapter outputs/exp1_baseline \
            --base Qwen/Qwen2.5-3B --out results/exp1_baseline.json
    """
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from code.dit_data import load_no_robots

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--base", required=True, help="Base model id")
    parser.add_argument("--model", default=None, help="Override --base (mirrors train.py --model)")
    # --out is the canonical name; --output is accepted as a synonym so
    # the smoke-test command (which uses --output) works as written.
    parser.add_argument("--out", "--output", dest="out", required=True, help="Path to write JSON results")
    parser.add_argument("--n-mem", type=int, default=100)
    parser.add_argument("--n-perturb", type=int, default=100)
    parser.add_argument("--n-arc", type=int, default=1172)
    parser.add_argument("--max-new", type=int, default=128, help="Max new tokens for memorization generation")
    parser.add_argument("--max-new-arc", type=int, default=4, help="Max new tokens for ARC generation (answers are single letters)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gen-batch-size", type=int, default=8, help="Batch size for ARC/memorization generation")
    parser.add_argument("--ppl-batch-size", type=int, default=16, help="Batch size for perplexity")
    parser.add_argument("--format-batch-size", type=int, default=8, help="Number of questions batched at once in format_robustness (each contributes 5 perturbations)")
    # New metric knobs (metrics 5-7)
    parser.add_argument("--wiki-ppl-n", type=int, default=0,
                        help="Limit wikitext PPL to first N examples. 0 = use all of test split.")
    parser.add_argument("--hellaswag-n", type=int, default=1000,
                        help="Limit HellaSwag eval to first N examples. 0 = use all 10042.")
    # Skip flags for each of the 7 metrics. The original 4 (ppl, lcs,
    # arc, format) were not present before but are added so smoke tests
    # can skip the slow existing metrics while exercising the new ones.
    parser.add_argument("--skip-ppl", action="store_true", help="Skip no_robots perplexity")
    parser.add_argument("--skip-lcs", action="store_true", help="Skip verbatim memorization (also skips diversity, which depends on its generations)")
    parser.add_argument("--skip-arc", action="store_true", help="Skip ARC-Challenge")
    parser.add_argument("--skip-format", action="store_true", help="Skip format robustness")
    parser.add_argument("--skip-wiki-ppl", action="store_true", help="Skip wikitext-103 PPL")
    parser.add_argument("--skip-hellaswag", action="store_true", help="Skip HellaSwag")
    parser.add_argument("--skip-diversity", action="store_true", help="Skip distinct-n + repetition-4 (does NOT skip the LCS generations; only the post-processing)")
    args = parser.parse_args()

    transformers_set_seed = __import__("transformers").set_seed
    transformers_set_seed(args.seed)

    # CLI --model wins over --base so smoke / orchestration scripts can
    # retarget eval at a different base without rewriting their --base arg.
    base_model = args.model if args.model is not None else args.base

    # Per-metric resumability: load any partial results from a previous
    # run of this same --out path, skip already-completed metrics, and
    # persist after every metric so a crash loses at most the metric in
    # progress rather than the whole eval.
    results: dict[str, Any] = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                results = json.load(f)
            if not isinstance(results, dict):
                results = {}
        except (json.JSONDecodeError, OSError):
            results = {}
    if not results:
        results = {"adapter": args.adapter, "base": base_model}
    else:
        results.setdefault("adapter", args.adapter)
        results.setdefault("base", base_model)
        done = [k for k in results if k not in ("adapter", "base")]
        if done:
            print(f"[eval] Resuming from existing results at {args.out}")
            print(f"[eval] Already-completed metrics: {done}")

    def _save_results() -> None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"[eval] Loading base model {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Critical: required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    print(f"[eval] Attaching adapter from {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print("[eval] Loading no_robots train + test")
    train_data, test_data = load_no_robots(cache_dir=args.cache_dir)

    if "perplexity" not in results:
        if args.skip_ppl:
            print("[eval] Perplexity: skipped (--skip-ppl)")
        else:
            print("[eval] Perplexity")
            results["perplexity"] = eval_perplexity(
                model, tokenizer, test_data, batch_size=args.ppl_batch_size
            )
            _save_results()
    else:
        print("[eval] Perplexity: cached, skipping")

    if "memorization" not in results:
        if args.skip_lcs:
            print("[eval] Verbatim memorization: skipped (--skip-lcs)")
        else:
            print("[eval] Verbatim memorization")
            # Reuse these generations for the diversity metric so we
            # don't have to re-run inference. Only build the sink when
            # the diversity metric is actually requested.
            generations_sink: list[str] = [] if not args.skip_diversity else []
            results["memorization"] = eval_verbatim_memorization(
                model,
                tokenizer,
                train_data,
                n=args.n_mem,
                max_new_tokens=args.max_new,
                seed=args.seed,
                batch_size=args.gen_batch_size,
                generations_sink=generations_sink,
            )
            _save_results()
            if not args.skip_diversity and generations_sink:
                print("[eval] Diversity (reuses LCS generations)")
                results["diversity"] = _diversity_eval(generations_sink)
                _save_results()
    else:
        print("[eval] Memorization: cached, skipping")

    if "arc" not in results:
        if args.skip_arc:
            print("[eval] ARC-Challenge: skipped (--skip-arc)")
        else:
            print("[eval] ARC-Challenge")
            results["arc"] = eval_arc_challenge(
                model,
                tokenizer,
                n=args.n_arc,
                max_new_tokens=args.max_new_arc,
                cache_dir=args.cache_dir,
                batch_size=args.gen_batch_size,
            )
            _save_results()
    else:
        print("[eval] ARC: cached, skipping")

    if "format_robustness" not in results:
        if args.skip_format:
            print("[eval] Format robustness: skipped (--skip-format)")
        else:
            print("[eval] Format robustness")
            results["format_robustness"] = eval_format_robustness(
                model,
                tokenizer,
                n=args.n_perturb,
                n_perturbations=5,
                max_new_tokens=64,
                seed=args.seed,
                cache_dir=args.cache_dir,
                batch_size_questions=args.format_batch_size,
            )
            _save_results()
    else:
        print("[eval] Format robustness: cached, skipping")

    # ---- new metrics 5-7 (not gated by the resumability cache; these
    # are cheap and idempotent, so re-running them is fine and skipping
    # them via a flag is the user's escape hatch) ----------------------

    if "wiki_ppl" not in results:
        if args.skip_wiki_ppl:
            print("[eval] Wikitext-103 PPL: skipped (--skip-wiki-ppl)")
        else:
            print("[eval] Wikitext-103 PPL")
            results["wiki_ppl"] = _wiki_ppl_eval(
                model,
                tokenizer,
                batch_size=args.ppl_batch_size,
                max_examples=args.wiki_ppl_n,
                max_length=1024,
                cache_dir=args.cache_dir,
            )
            _save_results()
    else:
        print("[eval] Wikitext-103 PPL: cached, skipping")

    if "hellaswag" not in results:
        if args.skip_hellaswag:
            print("[eval] HellaSwag: skipped (--skip-hellaswag)")
        else:
            print("[eval] HellaSwag")
            results["hellaswag"] = _hellaswag_eval(
                model,
                tokenizer,
                n=args.hellaswag_n,
                batch_size=args.gen_batch_size,
                cache_dir=args.cache_dir,
            )
            _save_results()
    else:
        print("[eval] HellaSwag: cached, skipping")

    print(f"[eval] Wrote {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
