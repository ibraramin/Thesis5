"""
dit_loss.py
===========
Core of the DITTP thesis experiment.

Implements the Decoupled Information-Theoretic Token Priority loss.
The standard HF Trainer reduces cross-entropy across the token dimension
automatically. We override that reduction so every token position gets
its own scalar weight derived from one of three signals:

  1. prompt_weight  -- a static scalar for prompt (instruction) tokens
  2. tfidf_dict     -- a {token_id: weight} lookup from corpus statistics
  3. entropy_mode   -- per-token Shannon entropy of the model's own
                       predictive distribution (requires float32 cast)

Loss formula:
    weight[t] = 0                           if label[t] == -100
              = prompt_weight               if position_type[t] == 1
              = tfidf[token_id[t]]          if position_type[t] == 2 and tfidf mode
              = entropy(logits[t])          if position_type[t] == 2 and entropy mode
              = 1.0                         if position_type[t] == 2 and neither

    loss = (per_token_ce * weight).sum() / weight.sum().clamp(min=1.0)
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


# ---------------------------------------------------------------------------
# Pure loss math
# ---------------------------------------------------------------------------

def per_token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Unreduced cross-entropy at every (batch, position) pair.

    HF Trainer's default CE applies a shift internally; we mirror that by
    predicting position t from logits at t-1. This keeps the loss aligned
    with the standard causal-LM objective so that vanilla experiments and
    DITTP experiments differ only in the per-token weighting, not in the
    underlying prediction target.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())


def shannon_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy over the vocabulary dimension.

    The float32 cast BEFORE softmax is mandatory: bf16 softmax on logits
    with large magnitude produces -inf log-probs which then become 0*log(0)
    -> NaN. Casting up first kills that branch entirely.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-9))
    return -(probs * log_probs).sum(dim=-1)


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

def build_position_weights(
    labels: torch.Tensor,
    position_type: torch.Tensor,
    prompt_weight: float,
    tfidf_tensor: torch.Tensor | None,
    entropy_mode: bool,
    logits: torch.Tensor | None,
) -> torch.Tensor:
    """Assemble the per-token weight tensor used to scale the loss.

    labels         : (B, S)        -- -100 where loss is masked
    position_type  : (B, S)        -- 0=masked, 1=prompt, 2=response
    tfidf_tensor   : (V,) or None  -- precomputed per-vocab-id weight
    entropy_mode   : bool          -- if True, override response weight
                                      with live Shannon entropy
    logits         : (B, S, V)     -- required iff entropy_mode is True
    """
    weights = torch.zeros_like(labels, dtype=torch.float32)

    # Prompt positions: constant scalar weight, same value for every token.
    prompt_mask = position_type == 1
    weights[prompt_mask] = float(prompt_weight)

    # Response positions: choose one of three sources.
    response_mask = position_type == 2

    if entropy_mode:
        if logits is None:
            raise ValueError("entropy_mode=True requires logits in build_position_weights")
        # We need entropy aligned with the CE shift: predict t from t-1.
        shifted_logits = logits[..., :-1, :].contiguous().float()
        shifted_response = response_mask[..., 1:].contiguous()
        entropy = shannon_entropy(shifted_logits)
        # Where the label is masked at the shifted position (i.e. the
        # corresponding original position is not a response), force weight
        # to 0 so CE masking still wins.
        per_position = torch.where(shifted_response, entropy, torch.zeros_like(entropy))
        # Map the shifted response mask back to the (B, S) layout, then
        # OVERLAY entropy on those positions. Crucially we do not
        # overwrite `weights` -- the prompt_weight that was set on
        # weights[prompt_mask] above must survive into the entropy branch
        # or exp4_entropy silently degenerates to "no prompt anchor".
        shifted_response_full = torch.zeros_like(labels, dtype=torch.bool)
        shifted_response_full[..., 1:] = shifted_response
        weights = torch.where(shifted_response_full, per_position, weights)
    else:
        if tfidf_tensor is not None:
            ids = labels.clamp(min=0)
            tfidf_w = tfidf_tensor.to(labels.device)[ids]
            weights = torch.where(response_mask, tfidf_w, weights)
        else:
            weights = torch.where(response_mask, torch.ones_like(weights), weights)

    return weights


# ---------------------------------------------------------------------------
# DITTP-aware Trainer
# ---------------------------------------------------------------------------

class DITTPComputeLoss:
    """Helper that holds experiment config and computes the weighted loss.

    We keep this separate from the Trainer subclass so the math is testable
    in isolation and so the Trainer subclass stays a thin override.
    """

    def __init__(
        self,
        prompt_weight: float,
        tfidf_tensor: torch.Tensor | None,
        entropy_mode: bool,
    ) -> None:
        self.prompt_weight = float(prompt_weight)
        self.tfidf_tensor = tfidf_tensor
        self.entropy_mode = bool(entropy_mode)

    def __call__(
        self,
        model_outputs,
        labels: torch.Tensor,
        position_type: torch.Tensor,
    ) -> torch.Tensor:
        logits = model_outputs.logits
        per_token_ce = per_token_cross_entropy(logits, labels)

        weights = build_position_weights(
            labels=labels,
            position_type=position_type,
            prompt_weight=self.prompt_weight,
            tfidf_tensor=self.tfidf_tensor,
            entropy_mode=self.entropy_mode,
            logits=logits if self.entropy_mode else None,
        )

        # Slice weights to match the shifted CE tensor (predict t from t-1).
        weights_shifted = weights[..., 1:].contiguous()
        ce_shifted = per_token_ce

        denom = weights_shifted.sum().clamp(min=1.0)
        return (ce_shifted * weights_shifted).sum() / denom


class DITTPTrainer(Trainer):
    """HF Trainer subclass that calls DITTPComputeLoss instead of internal CE.

    compute_loss is overridden to (a) accept the extra position_type and
    tfidf tensors delivered by DITCollator and (b) hand the model outputs
    off to DITTPComputeLoss. The standard reduction path is bypassed.
    """

    def __init__(
        self,
        *args,
        prompt_weight: float,
        tfidf_tensor: torch.Tensor | None,
        entropy_mode: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dit_loss = DITTPComputeLoss(
            prompt_weight=prompt_weight,
            tfidf_tensor=tfidf_tensor,
            entropy_mode=entropy_mode,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override of Trainer.compute_loss that respects DITTP weights.

        We pop the custom tensors out of inputs so the underlying forward
        pass only sees the arguments the base model expects.
        """
        labels = inputs.pop("labels")
        position_type = inputs.pop("position_type")
        # weight_tensor was precomputed by the collator and is informational
        # for logging; the authoritative weights are recomputed here so that
        # entropy mode (which needs live logits) works correctly.
        inputs.pop("weight_tensor", None)

        outputs = model(**inputs)
        loss = self._dit_loss(outputs, labels, position_type)
        return (loss, outputs) if return_outputs else loss
