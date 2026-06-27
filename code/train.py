"""
train.py
========
Main entry point for a single DITTP experiment.

Pipeline:
  1. CLI parses --config, --experiment, and --smoke.
  2. YAML supplies training hyperparams and the per-experiment overrides.
  3. We load the tokenizer, build the (V,) TF-IDF lookup if needed, and
     create a DITCollator bound to the current prompt_weight.
  4. The base model is loaded with flash_attention_2 and bf16; LoRA is
     applied via PEFT; gradient checkpointing is enabled (with
     enable_input_require_grads so LoRA grads flow through).
  5. A DITTPTrainer runs the training loop and saves only the LoRA adapter.

Smoke mode halves the dataset to 50 examples, uses 1 epoch, batch=2, and
disables gradient checkpointing for speed. The same loss math runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Make sibling-module imports work regardless of the cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dit_data import DITCollator, build_tfidf_tensor, load_no_robots  # noqa: E402
from dit_loss import DITTPTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# YAML / experiment resolution
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def select_experiment(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for exp in cfg.get("experiments", []):
        if exp.get("name") == name:
            return exp
    raise KeyError(f"Experiment '{name}' not found in config")


# ---------------------------------------------------------------------------
# Half-epoch checkpoint callback
# ---------------------------------------------------------------------------


class HalfEpochCheckpoint(TrainerCallback):
    """Save the LoRA adapter at the 0.5-epoch mark.

    HF Trainer's built-in save_strategy='steps' would technically work too,
    but coupling it to a single training run is cleaner with an explicit
    callback: the threshold stays the same no matter how the batch size or
    gradient accumulation is later retuned.
    """

    def __init__(self, save_root: str) -> None:
        self.save_root = save_root
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._fired or state.max_steps <= 0:
            return control
        # state.epoch is fractional; trigger at >= 0.5 if num_epochs >= 1.
        if (
            state.epoch is not None
            and state.num_train_epochs >= 1
            and state.epoch >= 0.5 * state.num_train_epochs
        ):
            os.makedirs(self.save_root, exist_ok=True)
            kwargs["model"].save_pretrained(os.path.join(self.save_root, "half_epoch"))
            self._fired = True
        return control


# ---------------------------------------------------------------------------
# Smoke mode overrides
# ---------------------------------------------------------------------------


def apply_smoke_overrides(cfg: dict[str, Any], exp: dict[str, Any]) -> None:
    """Mutate the configs in place to make a 50-sample, 1-epoch dry run."""
    cfg["training"]["num_train_epochs"] = 1
    cfg["training"]["per_device_train_batch_size"] = 2
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["gradient_checkpointing"] = False
    cfg["training"]["logging_steps"] = 1
    cfg["training"]["save_strategy"] = "no"
    exp["smoke"] = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--experiment", required=True, help="Experiment name from YAML")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model id from YAML (e.g. Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume from a checkpoint. Pass 'auto' (the default if the flag "
            "is present with no value) to auto-detect the latest "
            "checkpoint-XXX/ in output_dir, or pass an explicit path."
        ),
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Override the config max_train_samples (cap on training "
            "examples). Useful for small-scale pilot runs without editing "
            "the YAML. 0 or unset means use the full split."
        ),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help=(
            "Override the config num_train_epochs. Useful for cheap 1-epoch "
            "pilot runs without editing the YAML."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=["full", "micro"],
        default="full",
        help=(
            "Training profile. 'micro' caps the train split at "
            "cfg.micro.max_train_samples and num_train_epochs to "
            "cfg.micro.num_train_epochs for cheap direction-finding runs. "
            "'full' uses the experiment's normal training budget."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp = select_experiment(cfg, args.experiment)

    # CLI --model overrides the YAML 'model' key so the same config can
    # target multiple base models (3B vs 1.5B smoke test, etc.).
    if args.model is not None:
        cfg["model"] = args.model

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if args.smoke:
        print("[train] SMOKE mode active")
        apply_smoke_overrides(cfg, exp)

    # Micro profile: cap the train split and force 1 epoch. Comes from the
    # top-level `micro:` block in config.yaml. Applied AFTER smoke so a
    # `--smoke --profile micro` invocation keeps the 50-sample smoke
    # budget and just adds an explicit epoch=1 (already the smoke default).
    if args.profile == "micro":
        micro_cfg = cfg.get("micro", {}) or {}
        micro_n = int(micro_cfg.get("max_train_samples", 2000))
        micro_epochs = int(micro_cfg.get("num_train_epochs", 1))
        cfg["max_train_samples"] = micro_n
        cfg["training"]["num_train_epochs"] = micro_epochs
        exp["profile"] = "micro"
        print(
            f"[train] PROFILE=micro: {micro_n} examples, "
            f"{micro_epochs} epoch"
        )

    model_name: str = cfg["model"]
    output_dir = os.path.join("outputs", exp["name"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"[train] Experiment: {exp['name']}")
    print(f"[train] Output dir: {output_dir}")
    print(f"Loading model: {model_name}")
    print(f"[train] Seed: {seed}")
    print(
        f"[train] Config: prompt_weight={exp['prompt_weight']} "
        f"tfidf_mode={exp['tfidf_mode']} entropy_mode={exp['entropy_mode']}"
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("[train] Loading no_robots ...")
    train_data, _ = load_no_robots()
    # Pilot mode: cap the train split. CLI flag overrides config.
    max_train_samples = int(cfg.get("max_train_samples", 0))
    if args.max_train_samples is not None:
        max_train_samples = int(args.max_train_samples)
    if exp.get("smoke"):
        train_data = train_data[:50]
    elif max_train_samples > 0:
        train_data = train_data[:max_train_samples]
        print(
            f"[train] PILOT mode: capped train split at "
            f"{max_train_samples} examples"
        )
    print(f"[train] Train examples: {len(train_data)}")

    tfidf_tensor = None
    if exp.get("tfidf_mode"):
        print("[train] Building TF-IDF tensor (cached after first build) ...")
        tfidf_tensor = build_tfidf_tensor(
            tokenizer=tokenizer,
            train_split=train_data,
            model_name=model_name,
            cache_dir=cfg.get("tfidf_cache_dir", "./cache"),
            direction=str(exp.get("tfidf_direction", "standard")),
        )
        print(f"[train] TF-IDF tensor shape: {tuple(tfidf_tensor.shape)}")

    collator = DITCollator(
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 1024)),
        prompt_weight=float(exp["prompt_weight"]),
        tfidf_tensor=tfidf_tensor,
        im_mode=bool(exp.get("im_mode", False)),
    )

    train_ds = Dataset.from_list(train_data)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("[train] Loading base model in bf16 with flash_attention_2 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False  # required when using grad checkpointing

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    use_grad_ckpt = bool(cfg["training"].get("gradient_checkpointing", True))
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
        # Required for grad checkpointing + LoRA: lets the frozen base
        # model's inputs flow gradients into the LoRA adapters.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # torch.compile was tried here in an earlier revision; reverted after
    # smoke testing showed it (a) is a net slowdown at batch=4 due to
    # graph breaks from scalar logging and (b) silently breaks the LoRA
    # resume path because it wraps the model in OptimizedModule *before*
    # the manual adapter load runs. Eager mode is correct and within
    # budget; if compile becomes useful again, it must be applied AFTER
    # the resume adapter load so the inner PEFT model is the target of
    # set_peft_model_state_dict.

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    training_args_cfg = dict(cfg["training"])
    # Translate the YAML "0.5 epoch" intent into a concrete save_steps.
    eff_batch = int(training_args_cfg["per_device_train_batch_size"]) * int(
        training_args_cfg["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, math.ceil(len(train_data) / eff_batch))
    if float(training_args_cfg.get("save_steps", 0)) == 0.5:
        training_args_cfg["save_steps"] = max(1, steps_per_epoch // 2)
        training_args_cfg["save_strategy"] = "steps"
    # Build the actual TrainingArguments object.
    num_epochs = int(training_args_cfg["num_train_epochs"])
    if args.num_train_epochs is not None:
        num_epochs = int(args.num_train_epochs)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(
            training_args_cfg["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training_args_cfg["gradient_accumulation_steps"]
        ),
        learning_rate=float(training_args_cfg["learning_rate"]),
        num_train_epochs=num_epochs,
        lr_scheduler_type=training_args_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training_args_cfg.get("warmup_ratio", 0.05)),
        bf16=bool(training_args_cfg.get("bf16", True)),
        gradient_checkpointing=False,  # we already enabled it on the model
        logging_steps=int(training_args_cfg.get("logging_steps", 10)),
        save_strategy=training_args_cfg.get("save_strategy", "epoch"),
        save_steps=int(training_args_cfg.get("save_steps", 500)),
        save_total_limit=int(training_args_cfg.get("save_total_limit", 5)),
        report_to=training_args_cfg.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=False,  # collator emits position_type etc.
        dataloader_num_workers=int(training_args_cfg.get("dataloader_num_workers", 2)),
        optim=training_args_cfg.get("optim", "adamw_torch"),
        max_grad_norm=float(training_args_cfg.get("max_grad_norm", 1.0)),
    )

    trainer = DITTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        prompt_weight=float(exp["prompt_weight"]),
        tfidf_tensor=tfidf_tensor,
        entropy_mode=bool(exp["entropy_mode"]),
        loss_mode=str(exp.get("loss_mode", "weighted_mean")),
        im_mode=bool(exp.get("im_mode", False)),
    )

    # Re-add the half-epoch callback in non-smoke mode.
    callbacks = []
    if not exp.get("smoke") and float(cfg["training"].get("save_steps", 0)) == 0.5:
        callbacks.append(HalfEpochCheckpoint(output_dir))
    if callbacks:
        trainer.add_callback(callbacks)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # Resume logic: --resume-from-checkpoint without value (or "auto")
    # auto-detects the most recent checkpoint-XXX/ in output_dir. If the
    # flag is absent, training starts fresh.
    resume_ckpt: str | bool = False
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "auto":
            candidates = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
            if candidates:
                resume_ckpt = candidates[-1]
                print(f"[train] Auto-detected resume checkpoint: {resume_ckpt}")
            else:
                print(
                    "[train] --resume-from-checkpoint auto: no checkpoint found, starting fresh"
                )
        elif os.path.isdir(args.resume_from_checkpoint):
            resume_ckpt = args.resume_from_checkpoint
            print(f"[train] Resuming from explicit checkpoint: {resume_ckpt}")
        else:
            print(
                f"[train] WARNING: --resume-from-checkpoint value "
                f"'{args.resume_from_checkpoint}' is not a directory; "
                "starting fresh"
            )

    print("[train] Starting training ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Wall-clock timing spans both resume and fresh paths below; the two
    # `trainer.train(...)` invocations are the only thing that can
    # dominate the runtime of this script.
    t0 = time.time()
    if resume_ckpt:
        # LoRA resume workaround. HF Trainer's _load_from_checkpoint
        # calls load_sharded_checkpoint, which requires
        # model.safetensors.index.json. PEFT/LoRA checkpoints only have
        # adapter_model.safetensors (no index), so the standard path
        # always raises "Can't find a checkpoint index". We catch that
        # specific error and fall back to a manual adapter load plus
        # trainer_state.json restore. Optimizer and scheduler restart
        # fresh -- acceptable for smoke validation, since the model
        # weights (the part that matters) are preserved.
        try:
            trainer.train(resume_from_checkpoint=resume_ckpt)
        except ValueError as e:
            if "Can't find a checkpoint index" not in str(e):
                raise
            print(
                f"[train] HF Trainer auto-resume failed for LoRA ({e}); "
                "falling back to manual adapter load"
            )
            assert isinstance(resume_ckpt, str)  # narrow for type checker
            from safetensors.torch import load_file  # noqa: E402
            from peft import set_peft_model_state_dict  # noqa: E402

            adapter_path = os.path.join(resume_ckpt, "adapter_model.safetensors")
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(
                    f"Expected LoRA adapter at {adapter_path} but it is missing"
                )
            adapter_state = load_file(adapter_path)
            set_peft_model_state_dict(model, adapter_state)
            with open(os.path.join(resume_ckpt, "trainer_state.json")) as f:
                saved_state = json.load(f)
            trainer.state.global_step = int(saved_state.get("global_step", 0))
            trainer.state.epoch = float(saved_state.get("epoch", 0.0))
            if "max_steps" in saved_state:
                trainer.state.max_steps = int(saved_state["max_steps"])
            print(
                f"[train] Resumed model from {resume_ckpt} "
                f"(step={trainer.state.global_step}, "
                f"epoch={trainer.state.epoch:.2f}); "
                "optimizer/scheduler restart fresh"
            )
            trainer.train()
    else:
        trainer.train()
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[train] Peak VRAM: {peak_gb:.2f} GB")
    elapsed = time.time() - t0
    total_steps = int(getattr(trainer.state, "global_step", 0))
    sec_per_step = elapsed / max(1, total_steps)
    print(
        f"[train] Wall-clock: {elapsed:.1f}s "
        f"({sec_per_step:.3f} s/step over {total_steps} steps)"
    )

    # ------------------------------------------------------------------
    # Save final adapter
    # ------------------------------------------------------------------
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Persist a small summary so downstream eval knows the experiment shape.
    summary = {
        "experiment": exp["name"],
        "model": model_name,
        "prompt_weight": float(exp["prompt_weight"]),
        "tfidf_mode": bool(exp["tfidf_mode"]),
        "tfidf_direction": str(exp.get("tfidf_direction", "standard")),
        "entropy_mode": bool(exp["entropy_mode"]),
        "loss_mode": str(exp.get("loss_mode", "weighted_mean")),
        "im_mode": bool(exp.get("im_mode", False)),
        "smoke": bool(exp.get("smoke", False)),
        "profile": str(exp.get("profile", "full")),
        "steps_per_epoch": steps_per_epoch,
        "n_train": len(train_data),
        "seed": seed,
        "lora": lora_cfg,
    }
    with open(os.path.join(output_dir, "experiment.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Saved adapter to {final_path}")


if __name__ == "__main__":
    main()
