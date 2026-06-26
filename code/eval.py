"""
eval.py
=======
Four deterministic evaluation metrics for DITTP-trained adapters.

The functions all take a (model, tokenizer) pair. The caller is responsible
for merging the LoRA adapter into the base model beforehand; the eval
functions themselves do not touch PEFT. This keeps the eval math focused
on the metrics rather than on adapter plumbing.

Metrics:
  1. eval_perplexity           -- standard PPL over the no_robots test split
  2. eval_verbatim_memorization -- longest common substring vs. training refs
  3. eval_arc_challenge         -- exact-match A/B/C/D on ai2_arc
  4. eval_format_robustness     -- variance of next-token probs across perturbations
"""

from __future__ import annotations

import difflib
import json
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
    conventional causal-LM PPL formulation.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    nlls = []
    n_tokens = 0
    batch: list[dict] = []

    def flush() -> None:
        nonlocal nlls, n_tokens
        if not batch:
            return
        texts = [_build_chat_prompt(tokenizer, ex["prompt"]) + ex["response"] for ex in batch]
        enc = tokenizer(
            texts,
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

    for ex in test_data:
        batch.append(ex)
        if len(batch) >= batch_size:
            flush()
    flush()

    avg_nll = sum(nlls) / max(n_tokens, 1)
    return {"perplexity": float(torch.tensor(avg_nll).exp().item()), "n_tokens": int(n_tokens)}


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
) -> dict[str, float]:
    """Sample n training prompts, generate, and measure longest common
    substring with the gold response. We report mean and std across n.

    The inner loop batches generation calls (left-padded). batch_size=4
    is the conservative default for max_new_tokens=128; the CLI default
    is 8 (overrideable via --gen-batch-size) and the user can drop it
    back to 4 if the 3B model OOMs at batch=8 * 128 new tokens.
    """
    model.eval()
    rng = random.Random(seed)
    sample = rng.sample(train_data, k=min(n, len(train_data)))

    lengths: list[int] = []
    for start in range(0, len(sample), batch_size):
        batch_ex = sample[start:start + batch_size]
        prompts = [ex["prompt"] for ex in batch_ex]
        gens = _generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
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
    max_new_tokens: int = 64,
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

    from dit_data import load_no_robots

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--base", required=True, help="Base model id")
    parser.add_argument("--model", default=None, help="Override --base (mirrors train.py --model)")
    parser.add_argument("--out", required=True, help="Path to write JSON results")
    parser.add_argument("--n-mem", type=int, default=100)
    parser.add_argument("--n-perturb", type=int, default=100)
    parser.add_argument("--n-arc", type=int, default=1172)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gen-batch-size", type=int, default=8, help="Batch size for ARC/memorization generation")
    parser.add_argument("--ppl-batch-size", type=int, default=16, help="Batch size for perplexity")
    parser.add_argument("--format-batch-size", type=int, default=8, help="Number of questions batched at once in format_robustness (each contributes 5 perturbations)")
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
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    print(f"[eval] Attaching adapter from {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print("[eval] Loading no_robots train + test")
    train_data, test_data = load_no_robots(cache_dir=args.cache_dir)

    if "perplexity" not in results:
        print("[eval] Perplexity")
        results["perplexity"] = eval_perplexity(
            model, tokenizer, test_data, batch_size=args.ppl_batch_size
        )
        _save_results()
    else:
        print("[eval] Perplexity: cached, skipping")

    if "memorization" not in results:
        print("[eval] Verbatim memorization")
        results["memorization"] = eval_verbatim_memorization(
            model,
            tokenizer,
            train_data,
            n=args.n_mem,
            max_new_tokens=args.max_new,
            seed=args.seed,
            batch_size=args.gen_batch_size,
        )
        _save_results()
    else:
        print("[eval] Memorization: cached, skipping")

    if "arc" not in results:
        print("[eval] ARC-Challenge")
        results["arc"] = eval_arc_challenge(
            model,
            tokenizer,
            n=args.n_arc,
            max_new_tokens=64,
            cache_dir=args.cache_dir,
            batch_size=args.gen_batch_size,
        )
        _save_results()
    else:
        print("[eval] ARC: cached, skipping")

    if "format_robustness" not in results:
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

    print(f"[eval] Wrote {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
