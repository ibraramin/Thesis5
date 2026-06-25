# DITTP Thesis Methodology — Current Working Spec

**Title:** *Information-Theoretic Token Priority in Supervised Fine-Tuning: A Decoupled Loss Optimization Framework*

**Status:** Locked-in decisions, current as of Phase 1 validation run.

---

## 1. The Research Gap

Standard SFT applies a binary masking policy: prompt tokens get weight 0, response tokens get weight 1. This produces two known failure modes:

1. **Contextual degradation** — masking the prompt entirely starves the model of contextual regularization, hurting robustness to prompt perturbations.
2. **Gradient starvation** — applying identical weight to all response tokens lets frequent structural tokens (articles, conjunctions) saturate the gradient budget, drowning out rare, semantically critical tokens.

**Core hypothesis:** A *Decoupled Information-Theoretic Token Priority* (DITTP) objective — static low-magnitude prompt weight + corpus-derived TF-IDF on response — reduces verbatim memorization and improves OOD reasoning vs uniform SFT, with no extra forward passes.

**Contingency:** If static TF-IDF fails (BPE fragmentation breaks the signal), pivot to *epistemic priority* — use the model's own per-token predictive entropy as the response weight.

---

## 2. Experimental Design

Six experiment runs isolate the variable of interest. Each is one full SFT training with the same data, base model, and hyperparams; only the loss weighting changes.

| Run | Name | prompt_weight | response_weight | Isolates |
|---|---|---|---|---|
| 1a | exp1_baseline | 0.0 | 1.0 (uniform) | Reference SFT |
| 1b | exp1_low | 0.1 | 1.0 (uniform) | Low contextual anchor |
| 1c | exp1_high | 0.5 | 1.0 (uniform) | High contextual anchor |
| 2 | exp2_lexical | 0.0 | TF-IDF [0.5, 2.0] | Lexical priority only |
| 3 | exp3_combined | 0.1 | TF-IDF [0.5, 2.0] | **Full DITTP** |
| 4 | exp4_entropy | 0.1 | live entropy (float32) | Epistemic contingency |

**Attribution logic:**
- (1b, 1c) vs 1a → effect of prompt anchoring
- 2 vs 1a → effect of TF-IDF response weighting
- 3 vs (1a, 1b, 2) → combined DITTP effect
- 4 vs (3) → static lexical vs dynamic epistemic

**A priori expected outcomes (from original methodology):**
- 1b/1c improve format robustness variance over 1a
- 2 reduces verbatim memorization length over 1a
- 3 wins on the union of metrics (memorization, robustness, OOD)
- 4 is the fallback if 2/3 fail to differentiate from 1a

---

## 3. Loss Math (Canonical Formula)

For each token at position *t* in the sequence, given per-token cross-entropy `ce[t]` from the model:

```
weight[t] = 0                          if label[t] == -100  (masked: prompt or pad)
          = prompt_weight              if pos[t] == 1       (prompt)
          = tfidf[id[t]]               if pos[t] == 2 AND tfidf_mode
          = entropy(logits[t])         if pos[t] == 2 AND entropy_mode
          = 1.0                        if pos[t] == 2 AND neither mode

loss = (ce * weight).sum() / weight.sum().clamp(min=1.0)
```

**Important properties:**
- Predict position *t* from logits at *t-1* (standard CE shift) — keeps the math aligned with vanilla SFT so the only difference is the weighting.
- `weight.sum().clamp(min=1.0)` prevents division by zero when an entire batch is masked.
- The `entropy_mode` branch recomputes weights from live logits every step; the `tfidf_mode` branch uses a precomputed (V,) lookup tensor.

---

## 4. Technical Stack (Hardware-Fit)

| Component | Value | Justification |
|---|---|---|
| Hardware | RTX 3090, 24 GB VRAM, 20 h budget | Hard constraint |
| Base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Llama 2 architecture, low baseline gives DITTP headroom |
| Dataset | `HuggingFaceH4/no_robots` (9,500 train / 500 test) | Human-authored, no distillation contamination, single-turn instruction following |
| Precision | bfloat16 | Avoids fp16 dynamic loss scaling underflow |
| Attention | `flash_attention_2` | Memory + speed; standard for non-causal LM training |
| Optimizer | AdamW (torch) | Standard for LoRA |
| Learning rate | 2e-5 | Standard for LoRA on 1-3B models |
| LR scheduler | cosine | Smooth decay prevents end-of-training forgetting |
| Warmup ratio | 0.05 | Initial spike protection for low-rank matrices |
| Max grad norm | 1.0 | Standard clipping |
| LoRA rank | 16 | 1.13% of base params trainable |
| LoRA alpha | 32 | Standard 2× rank |
| LoRA target modules | q,k,v,o,gate,up,down (7) | All attention + MLP projections |
| LoRA dropout | 0.05 | Standard |
| Per-device batch | 4 | VRAM budget allows on TinyLlama |
| Grad accumulation | 4 | Effective batch 16 |
| Max length | 1024 tokens | Comfortably fits no_robots |
| Epochs | 3 | Convergence without overfitting on 9.5k examples |
| Gradient checkpointing | True | Reduces activation memory |

**Expected total compute:** ~63 min per experiment × 6 = ~6.3 h training + ~1.5 h eval = **~8 h within the 20 h budget**.

---

## 5. TF-IDF Construction

```
1. Fit TfidfVectorizer on the response column, analyzer = tokenizer's
   pre-tokenization (so lexical units = tokenizer's word-level units).
2. Per-term score = max tfidf across the corpus (one number per term).
3. Min-Max normalize into [0.5, 2.0]:
     normalized = 0.5 + 1.5 * (s - s_min) / (s_max - s_min)
4. Project to sub-words: each subword averages the scores of its
   parent words (after stripping BPE leading-space markers).
5. Subwords with no parent match → corpus median (1.0 default).
6. Result is a (V,) float tensor cached on disk, keyed on model name.
```

**Why [0.5, 2.0]:** the lower bound keeps frequent tokens from being zeroed out entirely (preserves gradient flow on common patterns); the upper bound caps the gradient magnitude on rare tokens so a handful of long-tail words cannot dominate the loss.

---

## 6. Entropy Branch (Contingency)

When `entropy_mode=True`, the response weight becomes the live Shannon entropy of the model's own predictive distribution at that position:

```
probs      = softmax(logits.float())              # cast BEFORE softmax
log_probs  = log(probs.clamp(min=1e-9))
entropy[t] = -(probs * log_probs).sum(dim=-1)
```

**Why float32 cast is mandatory:** bf16 softmax on logits with large magnitude can produce -inf log-probs; the resulting `0 * log(0)` → NaN blows up training. Casting up first kills that branch.

The entropy branch uses the same `(B, S, V)` logits tensor the model already produced for the forward pass — no extra forward pass is needed.

---

## 7. Evaluation Metrics (Four Deterministic Tests)

| # | Metric | What it measures | Higher is better? |
|---|---|---|---|
| 1 | Perplexity on no_robots test (500 examples) | Distribution match | Lower |
| 2 | Verbatim Sub-string Memorization Length (longest common substring) | Memorization vs synthesis | Lower |
| 3 | ARC-Challenge exact-match A/B/C/D (1,172 questions) | OOD reasoning | Higher |
| 4 | Format Robustness Variance (5 perturbations × N questions) | Robustness to prompt noise | Lower |

**Why deterministic:** avoids LLM-as-judge subjectivity. Each metric is a closed-form computation.

---

## 8. Methodology Shifts From Original (with Justification)

| Original Spec | Current Spec | Reason |
|---|---|---|
| `meta-llama/Llama-3.2-3B` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Llama-3.2 is gated (auth required). Qwen2.5-3B tried as alternative, rejected for high baseline (diminishing returns mask DITTP effect). TinyLlama chosen for low baseline + Llama 2 architecture. |
| GSM8K for OOD | ARC-Challenge | GSM8K is math-specific; ARC is broader OOD reasoning. Also: TinyLlama baseline on GSM8K is 1.4% (floor effect). |
| Half-epoch single checkpoint | Periodic save every 200 steps + resume | Operational — protects against vast.ai disconnects. |
| Save step sentinel `0.5` | Real value `200` | Periodic saves cover the half-epoch mark; no need for special-cased sentinel. |
| eval results saved once at end | Per-metric save (4 JSON checkpoints) | Operational — same reason. |

**None of these shifts alter the core DITTP framework.** The loss formula, the six-run experimental design, the four-metric evaluation, and the attribution logic are unchanged.

---

## 9. Pre-Run Verification (Smoke)

Before launching the full matrix:

- 50-sample, 1-epoch, batch=2, grad_accum=1, no grad checkpointing.
- Pass criteria: loss decreases below 1.95 within 25 steps, no NaN/Inf, peak VRAM < 20 GB, adapter serializes to disk.
- Reference baseline: `smangrul/tinyllama_lora_norobots` reported val loss 1.91 after 1 epoch on the same base model + dataset.

Status: **smoke passed (2026-06-26).**

---

## 10. Pipeline (Operational)

```bash
# 1. Clean smoke outputs
rm -rf outputs/exp1_low outputs/smoke_1.1B

# 2. Phase 1 validation (~63 min, full config)
python code/train.py --config code/config.yaml --experiment exp1_low

# 3. If interrupted, resume
python code/train.py --config code/config.yaml --experiment exp1_low --resume-from-checkpoint

# 4. Full 6-experiment matrix (~3.75h, will skip exp1_low)
python code/run_experiments.py
```

**Resilience layers:**
- `train.py` periodic checkpoint (every 200 steps, keeps 5 most recent) — worst-case loss on crash: ~5 min.
- `train.py --resume-from-checkpoint` auto-detects latest checkpoint.
- `run_experiments.py` orchestrator auto-resumes in-progress experiments.
- `eval.py` per-metric save (4 JSON checkpoints) — worst-case loss on crash: 1 metric (~3-4 min).

---

## 11. Thesis Defense Map

If the experiments produce the expected pattern, the defense narrative is:

> *Standard SFT applies uniform cross-entropy across response tokens, leading to gradient starvation on rare semantic tokens and contextual drift on prompt understanding. We proposed a Decoupled Information-Theoretic Token Priority (DITTP) loss that (a) applies a small static weight to prompt tokens as a contextual anchor, and (b) reweights response tokens by their corpus TF-IDF score. Across 6 controlled runs on a 1.1B base model with 9,500 instruction examples, DITTP [showed/did not show] significant gains in verbatim memorization length and OOD reasoning compared to uniform SFT, with no additional training cost. In the contingency case, dynamic entropy weighting provided a parallel but distinct result, supporting the case that the static lexical signal was the operative variable.*

The thesis is defensible in *both* directions (DITTP wins or DITTP does not differentiate from baseline). The contingency ensures an answer either way.
