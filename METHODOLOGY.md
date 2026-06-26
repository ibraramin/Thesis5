# DITTP Thesis Methodology — Current Working Spec

**Title:** *Information-Theoretic Token Priority in Supervised Fine-Tuning: A Decoupled Loss Optimization Framework*

**Status:** Locked-in decisions as of 2026-06-26. Phase 0 (pilot, 6 runs at 5,000 ex × 1 ep) complete. Full-scale 2-config run (baseline + exp3_combined at 9,500 ex × 3 ep) in progress.

---

## 1. The Research Gap

Standard SFT applies a binary masking policy: prompt tokens get weight 0, response tokens get weight 1. This produces two known failure modes:

1. **Contextual degradation** — masking the prompt entirely starves the model of contextual regularization, hurting robustness to prompt perturbations.
2. **Gradient starvation** — applying identical weight to all response tokens lets frequent structural tokens (articles, conjunctions) saturate the gradient budget, drowning out rare, semantically critical tokens.

**Core hypothesis:** A *Decoupled Information-Theoretic Token Priority* (DITTP) objective — static low-magnitude prompt weight + corpus-derived TF-IDF on response — reduces verbatim memorization and improves OOD reasoning vs uniform SFT, with no extra forward passes.

**Contingency:** If static TF-IDF fails (BPE fragmentation breaks the signal), pivot to *epistemic priority* — use the model's own per-token predictive entropy as the response weight.

---

## 2. Experimental Design

**Two experiment runs isolate the primary DITTP hypothesis.** Each is one full SFT training (3 epochs × 9,500 examples) with the same data, base model, and hyperparams; only the loss weighting changes.

| Run | Name | prompt_weight | response_weight | Isolates |
|---|---|---|---|---|
| 1a | exp1_baseline | 0.0 | 1.0 (uniform) | Reference SFT |
| 3 | exp3_combined | 0.1 | TF-IDF [0.5, 2.0] | **Full DITTP** |

**Attribution logic (full-scale pair):**
- exp3_combined vs exp1_baseline → DITTP effect (union of prompt anchoring + TF-IDF response reweighting)

**Why the 2-config pair, not 6:** A 6-run pilot matrix at 1/3 scale (5,000 examples × 1 epoch) was run to validate the code paths and surface confounders. An independent review of the pilot loss curves found that 4 of the 6 runs (exp1_baseline, exp1_low, exp2_lexical, exp3_combined) were statistically indistinguishable on training loss (|t|<1.8 vs an estimated noise floor of 0.05–0.07). The pilot matrix is preserved as preliminary ablation; the full-scale pair is a clean DITTP-vs-SFT test of the core hypothesis within the 20 h budget. See §9 for pilot results.

**A priori expected outcomes (2-config):**
- exp3_combined improves at least one of: (a) reduced verbatim memorization length, (b) improved OOD reasoning (ARC-Challenge), (c) reduced format-robustness variance, vs exp1_baseline.
- If no metric differentiates, the thesis supports the null: the effect size of the DITTP framework is below the detection limit of this configuration (1.1B base, 9.5k examples, 3 epochs).

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

**Pilot finding (footnote, exp1_high denominator inflation):** When `prompt_weight=0.5` and the prompt-token count exceeds the response-token count on a sequence (no_robots turns often have longer instructions than responses, since responses are concise), `weight.sum()` is inflated by the prompt contribution and the per-token loss is deflated by an estimated **~20–25%** relative to runs where prompts are masked. This is a **denominator artifact of the loss formula**, not a true loss reduction. The exp1_high pilot final loss (1.5074) reflects this deflation; its true loss-equivalent is closer to ~1.8. Direct numerical comparison of exp1_high training loss to other runs is not meaningful; only the eval metrics are apples-to-apples. exp1_high is preserved in the pilot writeup for completeness but is not part of the full-scale pair.

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
| Max grad norm | 1.0 | Standard clipping (see §6 for exp4_entropy pathology) |
| LoRA rank | 16 | 1.13% of base params trainable |
| LoRA alpha | 32 | Standard 2× rank |
| LoRA target modules | q,k,v,o,gate,up,down (7) | All attention + MLP projections |
| LoRA dropout | 0.05 | Standard |
| Per-device batch | 4 | VRAM budget allows on TinyLlama |
| Grad accumulation | 4 | Effective batch 16 |
| Max length | 1024 tokens | Comfortably fits no_robots |
| Epochs | 3 | Convergence without overfitting on 9.5k examples |
| Gradient checkpointing | True | Reduces activation memory |

**Expected total compute (full-scale pair):** ~2.5–3 h per experiment × 2 = ~5–6 h training + ~30 min eval (after fix-1 batched eval.py) = **~6–7 h within the 20 h budget**, leaving ~13 h headroom for 3-seed follow-up runs and sensitivity panels.

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

**Pilot finding (grad-norm pathology):** exp4_entropy exhibited severe gradient norm pathology at 1-epoch/5,000-ex scale. Across 31 logging steps in the pilot:

| Metric | exp4_entropy | Other 5 runs |
|---|---|---|
| Mean grad norm | **1.495** | 0.10–0.30 |
| Max grad norm | **5.45** | ≤0.6 |
| Clip events (max_grad_norm=1.0) | **21/31** | 0/31 |

Two consequences for the full-scale design:

1. **exp3_combined (TF-IDF) is the primary DITTP run, not exp4_entropy.** A 1-config DITTP-vs-SFT comparison using exp4_entropy would be dominated by the grad-clip confound, not the loss-weighting effect.
2. **Any future re-run of exp4_entropy must use `max_grad_norm=5.0` or higher**, or remove clipping entirely. The entropy branch legitimately produces larger gradients than the uniform baseline because the loss is now weighted by a function of the model's own confidence (low-confidence tokens get upweighted → larger CE on hard tokens → larger gradients). Clipping at 1.0 then biases the effective learning rate downward in a way that the baseline does not experience.

---

## 7. Evaluation Metrics (Four Deterministic Tests)

| # | Metric | What it measures | Higher is better? |
|---|---|---|---|
| 1 | Perplexity on no_robots test (500 examples) | Distribution match | Lower |
| 2 | Verbatim Sub-string Memorization Length (longest common substring) | Memorization vs synthesis | Lower |
| 3 | ARC-Challenge exact-match A/B/C/D (1,172 questions) | OOD reasoning | Higher |
| 4 | Format Robustness Variance (5 perturbations × N questions) | Robustness to prompt noise | Lower |

**Why deterministic:** avoids LLM-as-judge subjectivity. Each metric is a closed-form computation.

**Eval speedup (Phase 0 → Phase 1):** `code/eval.py` was rewritten to batch all four metrics (`_generate_batch` for ARC and memorization, `_next_token_probs_batch` for format_robustness, larger batch for perplexity). Expected ~5–10× wall-clock reduction on the 24 GB GPU. Defaults: `--gen-batch-size 8 --ppl-batch-size 16 --format-batch-size 8`. **Sanity check (PPL identical within 1e-4 vs pre-change run) required on GPU before trusting batched numbers.**

---

## 8. Methodology Shifts From Original (with Justification)

| Original Spec | Current Spec | Reason |
|---|---|---|
| `meta-llama/Llama-3.2-3B` | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Llama-3.2 is gated (auth required). Qwen2.5-3B tried as alternative, rejected for high baseline (diminishing returns mask DITTP effect). TinyLlama chosen for low baseline + Llama 2 architecture. |
| GSM8K for OOD | ARC-Challenge | GSM8K is math-specific; ARC is broader OOD reasoning. Also: TinyLlama baseline on GSM8K is 1.4% (floor effect). |
| Half-epoch single checkpoint | Periodic save every 200 steps + resume | Operational — protects against vast.ai disconnects. |
| Save step sentinel `0.5` | Real value `200` | Periodic saves cover the half-epoch mark; no need for special-cased sentinel. |
| eval results saved once at end | Per-metric save (4 JSON checkpoints) | Operational — same reason. |
| **Six runs at full scale** | **Two runs (baseline + exp3_combined) at full scale; pilot matrix preserved as preliminary ablation** | Pilot showed 4 of 6 runs statistically indistinguishable on training loss (\|t\|<1.8 vs noise floor 0.05–0.07). Full-scale pair gives the cleanest DITTP-vs-SFT verdict within the 20 h budget. See §9. |

**None of these shifts alter the core DITTP framework.** The loss formula and the four-metric evaluation are unchanged. The 2-config pivot reduces ablation granularity: the full-scale pair cannot separately attribute the prompt-weight and TF-IDF contributions. That attribution is preserved in the pilot writeup.

---

## 9. Pilot Validation Results (Phase 0)

**Configuration:** 5,000 examples (random slice, seed=42) × 1 epoch, max_steps=312, total_flos=1.742e+16 across all runs. Same data slice, seed, base model, and hyperparams as the full-scale spec; only loss weighting varies.

**Final training loss (step 312):**

| Run | Final loss | Min loss | Argmin step | Notes |
|---|---|---|---|---|
| exp1_baseline | 1.9158 | 1.7299 | ~270 | Reference; standard convergence |
| exp1_low | 1.7950 | 1.6497 | ~270 | prompt_weight=0.1, response uniform |
| exp1_high | **1.5074** | 1.3556 | ~10 | Denominator-inflation artifact (see §3) |
| exp2_lexical | 1.8510 | 1.6697 | ~110 | TF-IDF only, prompt masked |
| exp3_combined | 1.7229 | 1.5751 | ~110 | Full DITTP at 1-epoch scale |
| exp4_entropy | 2.2696 | 2.1314 | ~270 | Grad-norm pathology (see §6) |

**Independent loss-curve review (oracle):** 4 of 6 runs (exp1_baseline, exp1_low, exp2_lexical, exp3_combined) are statistically indistinguishable at the per-epoch level (|t|<1.8 vs an estimated noise floor of 0.05–0.07). The 1-epoch "overfitting" pattern is reframed as model convergence at or before step 270 followed by post-convergence noise: argmin is at or before step 270 for 5 of 6 runs, and the gap from argmin to the final loss is within the noise band. exp1_high is excluded from this comparison (denominator artifact); exp4_entropy is excluded (grad-norm pathology).

**Pilot eval (partial, exp1_baseline, OLD eval.py):** Perplexity **7.98** (n_tokens=153,949), verbatim memorization **lcs_mean=21.17 std=24.60** (n=100). ARC-Challenge and format-robustness pending re-run with batched eval.py (fix-1).

**Pilot writeup status:** The 6-run matrix is preserved as preliminary ablation. The full-scale pair's apples-to-apples eval is the primary endpoint. The 1-epoch pilot is undertrained, not overfit; the full-scale 3-epoch run is the defensible test of the hypothesis.

---

## 10. Pre-Run Verification (Smoke)

Before launching the full matrix:

- 50-sample, 1-epoch, batch=2, grad_accum=1, no grad checkpointing.
- Pass criteria: loss decreases below 1.95 within 25 steps, no NaN/Inf, peak VRAM < 20 GB, adapter serializes to disk.
- Reference baseline: `smangrul/tinyllama_lora_norobots` reported val loss 1.91 after 1 epoch on the same base model + dataset.

Status: **smoke passed (2026-06-26).** Pilot (5,000 ex × 1 ep, 6 runs) passed (2026-06-26). Full-scale baseline run in progress (2026-06-26).

---

## 11. Pipeline (Operational)

```bash
# 1. Full-scale baseline (3 epochs × 9,500 examples, ~2.5-3 h)
python code/train.py --config code/config.yaml --experiment exp1_baseline

# 2. If interrupted, resume
python code/train.py --config code/config.yaml --experiment exp1_baseline --resume-from-checkpoint

# 3. Full-scale exp3_combined (3 epochs × 9,500 examples, ~2.5-3 h)
python code/train.py --config code/config.yaml --experiment exp3_combined

# 4. Apples-to-apples eval on both (4 metrics × 2 models, ~30 min total with fix-1)
python code/eval.py --adapter outputs/exp1_baseline/final --base TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --out results/exp1_baseline.json
python code/eval.py --adapter outputs/exp3_combined/final --base TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --out results/exp3_combined.json

# 5. (Conditional) 3-seed follow-up if primary endpoint differentiates
# See §12 for branching rule.
```

**Resilience layers:**
- `train.py` periodic checkpoint (every 200 steps, keeps 5 most recent) — worst-case loss on crash: ~5 min.
- `train.py --resume-from-checkpoint` auto-detects latest checkpoint; manual fallback for sharded checkpoint failures.
- `eval.py` per-metric save (4 JSON checkpoints) — worst-case loss on crash: 1 metric (~3-4 min).
- `eval.py` per-metric resumability (`"X" not in results` guards) — partial eval runs survive crashes.

---

## 12. Thesis Defense Map

**Primary narrative (DITTP wins on at least one metric):**
> *Standard SFT applies uniform cross-entropy across response tokens, leading to gradient starvation on rare semantic tokens and contextual drift on prompt understanding. We proposed a Decoupled Information-Theoretic Token Priority (DITTP) loss that (a) applies a small static weight to prompt tokens as a contextual anchor, and (b) reweights response tokens by their corpus TF-IDF score. Across a controlled 2-run experiment (baseline + DITTP) at 3 epochs × 9,500 instruction examples on a 1.1B base model, DITTP improved [metric X] by [delta] relative to uniform SFT with no additional training cost. A 6-run pilot matrix at 1/3 scale provides preliminary ablation: prompt anchoring alone (exp1_low) and TF-IDF alone (exp2_lexical) each moved the loss curve but neither was statistically separable from baseline at 1-epoch scale; only the union (exp3_combined) is recommended for full-scale evaluation.*

**Null narrative (DITTP does not differentiate from baseline):**
> *Across a controlled 2-run experiment (baseline + DITTP) at 3 epochs × 9,500 instruction examples on a 1.1B base model, DITTP did not differentiate from uniform SFT on any of the four evaluation metrics. The effect size of the DITTP framework, if any, is below the detection limit of this configuration. The pilot matrix's loss-curve findings (4 of 6 runs statistically indistinguishable) are consistent with this null result. A larger base model, a larger dataset, or a different TF-IDF construction (e.g., document-frequency-then-normalize rather than max-tfidf) may be required to detect the effect.*

The thesis is defensible in **both directions.** The contingency ensures an answer either way.
