
"""Train one LoRA "expert" per question type (score / noul / choice) on top of JPT.

Usage (from the repository root):
    python -m src.training.lora_jpt_expert --modality choice
    python -m src.training.lora_jpt_expert --modality score --batch-size 8 --run-name jpt-score-expert-v2
    python -m src.training.lora_jpt_expert --modality noul --datasets boolq paws
    python -m src.training.lora_jpt_expert --modality choice --max-steps 20 --no-save   # quick smoke test

What it does:
  1. Loads up to N train and validation rows per dataset (equal caps, so no task dominates).
     Test splits are never loaded.
  2. Refuses to train or validate on held-out sources, so they stay a clean zero-shot test set.
  3. Measures JPT on validation before training (epoch 0), so you can see what the LoRA adds.
  4. Trains a rank-64 LoRA on attention + MLP layers with warmup + cosine LR decay.
     Objective: hard-label Brier, or for score questions cross-entropy against Gaussian
     soft labels (ordinal smoothing). Choice questions get shuffled option order.
  5. Validates per dataset after every epoch (Brier, accuracy, ECE), keeps the adapter of
     the epoch with the best macro validation Brier, and writes a full run record.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from llm2jev.prompt import options_of
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

from src.training import lora_jpt_one_ds as base
from src.training.dataset import DATASET_ID, HF_TOKEN


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def to_binary_noul(ds):
    """mnli as a noul task: entailment -> 1, contradiction -> 0, neutral rows dropped."""
    question = json.dumps({
        "type": "noul",
        "instructions": "Based only on `premise`, is `hypothesis` definitely true?",
    })
    ds = ds.filter(lambda row: row["label"] in ("entailment", "contradiction"))
    return ds.map(lambda row: {
        "primitive": "noul",
        "question": question,
        "label": "1" if row["label"] == "entailment" else "0",
    })


# Configs built from another jev-bench config: name -> (source config, transform).
DERIVED = {
    "mnli_binary": ("mnli", to_binary_noul),
}


def source_of(name):
    """The jev-bench config a dataset is loaded from."""
    return DERIVED[name][0] if name in DERIVED else name


# Every usable config, grouped by question type.
# Source: https://huggingface.co/datasets/Praveenrajus/jev-bench
DATASETS_BY_MODALITY = {
    "score": (
        "sst5", "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity",
        "stsb", "measuring_hate_speech",
    ),
    "noul": (
        "boolq", "fever_evidence", "paws", "civil_comments", "sms_spam",
        "strategyqa_closed", "strategyqa_grounded", "mnli_binary",
    ),
    "choice": (
        "banking77", "clinc150", "massive", "ledgar", "go_emotions",
        "mmlu", "arc_challenge", "mnli",
    ),
}

# Never trained or validated on, so they stay a clean zero-shot test (the Jevify protocol).
# clinc150, fever_evidence and yelp5 are the benchmark's OOD sets. strategyqa_closed asks
# the same questions as strategyqa_grounded, so training on it would leak into that source.
HELD_OUT = frozenset({
    "clinc150", "arc_challenge", "yelp5", "measuring_hate_speech",
    "fever_evidence", "strategyqa_grounded", "strategyqa_closed",
})

# Default training mix: semantically coherent tasks per modality, no held-out sources.
#   choice: intent routing and clause classification (3 domains, 18 to 100 options)
#   noul:   reading comprehension, paraphrase and entailment (factual consistency only;
#           toxicity and spam heuristics are left out)
#   score:  quality and sentiment, where a higher level always means "better"
#           (verbosity and similarity measure unrelated features and are left out)
DEFAULT_DATASETS = {
    "choice": ("banking77", "massive", "ledgar"),
    "noul": ("boolq", "paws", "mnli_binary"),
    "score": ("helpsteer2_helpfulness", "sst5"),
}

# Rank 64 with alpha = 2 x rank, on attention, MLP and DeltaNet projections
# (see base.LORA_LEAF_NAMES); the vision tower and lm_head stay frozen.
LORA_SETTINGS = {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05, "bias": "none"}
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.05
SCORE_SIGMA = 1.0  # std of the Gaussian soft labels for score questions, in levels


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", choices=DATASETS_BY_MODALITY, default="score")
    p.add_argument("--datasets", nargs="+", help="jev-bench configs to train on (default: DEFAULT_DATASETS)")
    p.add_argument("--train-samples", type=int, default=2500, help="Max training rows per dataset")
    p.add_argument("--validation-samples", type=int, default=500, help="Max validation rows per dataset")
    p.add_argument("--model-id", default=base.MODEL_ID)
    p.add_argument("--revision", help="Exact model commit to load, for reproducible runs")
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO,
                   help="Share of optimizer updates spent on linear warmup before cosine decay")
    p.add_argument("--score-sigma", type=float, default=SCORE_SIGMA,
                   help="Gaussian label smoothing for score questions (0 = hard-label Brier)")
    p.add_argument("--epochs", type=int, default=base.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    p.add_argument("--validation-batch-size", type=int, default=base.VALIDATION_BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=base.SEED)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=importlib.util.find_spec("fla") is None,
                   help="Save VRAM at some speed cost; on by default when the fla kernels are missing")
    p.add_argument("--skip-baseline", action="store_true", help="Skip the epoch-0 validation before training")
    p.add_argument("--print-every-batches", type=int, default=base.PRINT_EVERY_BATCHES)
    p.add_argument("--max-steps", type=int, help="Cap batches per epoch, for a quick smoke test")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--run-name", help="Name for output files (default: modality plus a timestamp)")
    p.add_argument("--no-save", action="store_true", help="Skip saving weights; the run file is still written")
    args = p.parse_args()

    # Numbers that must be positive.
    for name in ("train_samples", "validation_samples", "batch_size", "validation_batch_size",
                 "grad_accum_steps", "epochs", "print_every_batches"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be positive")
    if args.learning_rate <= 0:
        p.error("--learning-rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        p.error("--warmup-ratio must be in [0, 1)")
    if args.score_sigma < 0:
        p.error("--score-sigma must not be negative")

    # Datasets: right question type, no duplicates, never a held-out source.
    args.datasets = args.datasets or list(DEFAULT_DATASETS[args.modality])
    if len(set(args.datasets)) != len(args.datasets):
        p.error("--datasets must not contain duplicates")
    wrong_type = set(args.datasets) - set(DATASETS_BY_MODALITY[args.modality])
    if wrong_type:
        p.error(f"Not {args.modality} configs: {sorted(wrong_type)}. "
                f"Choose from {', '.join(DATASETS_BY_MODALITY[args.modality])}")
    held_out_used = {name for name in args.datasets if source_of(name) in HELD_OUT}
    if held_out_used:
        p.error(f"{sorted(held_out_used)} are held out for zero-shot testing and can never be trained on.")

    args.run_name = args.run_name or f"jpt-{args.modality}-expert-{time.strftime('%Y%m%d-%H%M%S')}"
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_rows(name, split, limit, seed):
    """A random sample of up to `limit` rows from one jev-bench split (or derived config)."""
    ds = load_dataset(DATASET_ID, source_of(name), split=split, token=HF_TOKEN)
    if name in DERIVED:
        ds = DERIVED[name][1](ds)
    if len(ds) == 0:
        raise ValueError(f"{name}/{split} is empty")
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    if len(ds) < limit:
        print(f"{name}/{split}: using all {len(ds)} available rows (asked for {limit})", flush=True)
    return ds


def build_loaders(args, processor, tokenizer):
    """Load every dataset, render it with llm2jev's prompt and wrap it in DataLoaders.

    Training rows from all datasets are mixed into one shuffled loader. Validation
    stays separate per dataset, so results can be reported task by task.
    """
    train_raw, val_raw, counts = {}, {}, {}
    for i, name in enumerate(args.datasets):
        train_raw[name] = load_rows(name, "train", args.train_samples, args.seed + i)
        val_raw[name] = load_rows(name, "validation", args.validation_samples, args.seed + i)
        counts[name] = {"train": len(train_raw[name]), "validation": len(val_raw[name])}
        print(f"{name}: {counts[name]}", flush=True)

    # Last line of defence: no row of any split may come from a held-out source.
    sources = {s for ds in (*train_raw.values(), *val_raw.values()) for s in ds.unique("source")}
    sources |= {source_of(name) for name in args.datasets}
    assert not sources & HELD_OUT, f"Held-out sources in training data: {sorted(sources & HELD_OUT)}"

    # The answer labels (A, B, C, ...) must cover the question with the most options.
    questions = [q for ds in (*train_raw.values(), *val_raw.values()) for q in ds["question"]]
    max_options = max(len(options_of(json.loads(q))[0]) for q in questions)
    labels, token_ids = base.build_llm2jev_labels(processor, tokenizer, max_options)
    collate = base.JevBenchCollator(tokenizer, token_ids)

    # Training rows get shuffled option order (choice questions only), so the model
    # reads the options instead of memorizing positions. Validation keeps the original order.
    def render(ds, shuffle=False):
        return base.prepare_dataset(ds, processor, tokenizer, labels, args.modality,
                                    shuffle_options=shuffle, seed=args.seed)

    pin = base.DEVICE.type == "cuda"
    train_data = concatenate_datasets([render(ds, shuffle=True) for ds in train_raw.values()])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, pin_memory=pin)
    val_loaders = {
        name: DataLoader(render(ds), batch_size=args.validation_batch_size,
                         collate_fn=collate, pin_memory=pin)
        for name, ds in val_raw.items()
    }
    return train_loader, val_loaders, counts, max_options


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_processor_and_tokenizer(args):
    processor = AutoProcessor.from_pretrained(args.model_id, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def build_model(args):
    """Load JPT and attach a fresh LoRA adapter. Returns the model, its commit and the LoRA targets."""
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=base.DTYPE, attn_implementation="sdpa",
    ).to(base.DEVICE)
    model.config.use_cache = False
    commit = getattr(model.config, "_commit_hash", None)  # exact weights version, for the run file

    targets = base.find_lora_target_modules(model)
    print("LoRA targets per layer type:", dict(sorted(Counter(t.split(".")[-1] for t in targets).items())),
          flush=True)
    model = get_peft_model(model, LoraConfig(**LORA_SETTINGS, target_modules=targets))
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = targets if isinstance(targets, str) else sorted(targets)
    return model, commit, targets


def adapter_snapshot(model):
    """A CPU copy of the LoRA weights, to restore the best epoch later."""
    return {key: value.detach().to("cpu", copy=True) for key, value in get_peft_model_state_dict(model).items()}


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, params, args, epoch):
    """One pass over the mixed training data. Returns the epoch's mean loss, accuracy and timing."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    steps = min(len(loader), args.max_steps or len(loader))
    print(f"Epoch {epoch}: {len(loader.dataset)} mixed examples, {steps} batches", flush=True)

    # Running totals stay on the GPU: calling .item() every step would force a sync and slow training.
    loss_sum = torch.zeros((), device=base.DEVICE)
    correct_sum = torch.zeros((), device=base.DEVICE, dtype=torch.long)
    recent_loss = deque(maxlen=base.ROLLING_WINDOW)
    recent_correct = deque(maxlen=base.ROLLING_WINDOW)
    seen = tokens = examples_in_window = updates = 0
    if base.DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        n = batch["input_ids"].shape[0]
        result = base.forward_batch(model, batch, args.modality, score_sigma=args.score_sigma)

        # Gradient accumulation. Each micro-batch adds (mean loss x batch size); at the update,
        # gradients are divided by the examples in the window. That gives the exact per-example
        # mean even when the last micro-batch is smaller than the others.
        scaler.scale(result["loss"] * n).backward()
        examples_in_window += n
        if (step + 1) % args.grad_accum_steps == 0 or step + 1 == steps:
            scaler.unscale_(optimizer)
            for param in params:
                if param.grad is not None:
                    param.grad.div_(examples_in_window)
            torch.nn.utils.clip_grad_norm_(params, base.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            examples_in_window = 0
            updates += 1

        # Bookkeeping for the progress log (the training objective). detach() keeps old graphs
        # from piling up in memory.
        per_example_loss = result["per_example_objective"].detach()
        loss_sum += per_example_loss.sum()
        correct_sum += result["correct_mask"].sum()
        recent_loss.append(per_example_loss)
        recent_correct.append(result["correct_mask"])
        seen += n
        tokens += int(batch["attention_mask"].sum())

        if step == 0 or (step + 1) % args.print_every_batches == 0 or step + 1 == steps:
            if base.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rolling_loss = torch.cat(list(recent_loss))[-base.ROLLING_WINDOW:].mean().item()
            rolling_acc = torch.cat(list(recent_correct))[-base.ROLLING_WINDOW:].float().mean().item()
            eta_min = (steps - step - 1) * elapsed / (step + 1) / 60
            print(f"epoch={epoch} batch={step + 1}/{steps} updates={updates} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"loss={loss_sum.item() / seen:.5f} acc={correct_sum.item() / seen:.4f} "
                  f"rolling_loss={rolling_loss:.5f} rolling_acc={rolling_acc:.4f} "
                  f"examples/s={seen / elapsed:.1f} tokens/s={tokens / elapsed:.0f} "
                  f"ETA_min={eta_min:.1f}", flush=True)

    if base.DEVICE.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    return {
        "loss": loss_sum.item() / seen,
        "accuracy": correct_sum.item() / seen,
        "updates": updates,
        "seconds": round(time.perf_counter() - started, 1),
    }


def validate(model, loaders, modality):
    """Evaluate every validation set separately, then combine.

    "accuracy" pools all rows, so bigger datasets weigh more. The macro_* numbers
    average the per-dataset results, so every dataset counts equally.
    """
    per_dataset = {}
    for name, loader in loaders.items():
        per_dataset[name] = base.evaluate(model, loader, modality)
        print(f"  {name}: {per_dataset[name]}", flush=True)
    n = sum(r["n"] for r in per_dataset.values())
    overall = {
        "loss": sum(r["loss"] * r["n"] for r in per_dataset.values()) / n,
        "accuracy": sum(r["correct"] for r in per_dataset.values()) / n,
        "macro_accuracy": sum(r["accuracy"] for r in per_dataset.values()) / len(per_dataset),
        "macro_brier": sum(r["brier"] for r in per_dataset.values()) / len(per_dataset),
        "macro_ece": sum(r["ece"] for r in per_dataset.values()) / len(per_dataset),
        "n": n,
    }
    print(f"  overall: {overall}", flush=True)
    return {"datasets": per_dataset, "overall": overall}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def package_versions():
    versions = {}
    for package in ("torch", "transformers", "peft", "datasets", "llm2jev"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_run(args, record, model, processor, tokenizer):
    """Write the run record (always) and the LoRA + merged weights (unless --no-save)."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_text = json.dumps(record, indent=2) + "\n"
    record_path = args.output_dir / f"{args.run_name}-run.json"
    record_path.write_text(record_text, encoding="utf-8")

    # One summary line per run, without the per-epoch history, for quick comparisons.
    summary = {key: value for key, value in record.items() if key != "history"}
    with (args.output_dir / "lora_run_times.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Run record: {record_path}", flush=True)

    if args.no_save:
        return
    lora_dir = args.output_dir / f"{args.run_name}-lora"
    merged_dir = args.output_dir / f"{args.run_name}-merged"
    base.save_lora(model, processor, tokenizer, lora_dir)
    base.merge_and_save(model, processor, tokenizer, merged_dir)
    for directory in (lora_dir, merged_dir):
        (directory / "training_metadata.json").write_text(record_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if base.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name(), flush=True)
    score_objective = ("gaussian_soft_label_ce" if args.modality == "score" and args.score_sigma > 0
                       else "hard_label_brier")
    print(f"Run {args.run_name}: modality={args.modality}, datasets={args.datasets}, "
          f"lr={args.learning_rate}, warmup={args.warmup_ratio}, objective={score_objective}, "
          f"batch={args.batch_size}x{args.grad_accum_steps}, "
          f"dtype={base.DTYPE}, gradient_checkpointing={args.gradient_checkpointing}", flush=True)

    # 1. Data
    processor, tokenizer = load_processor_and_tokenizer(args)
    train_loader, val_loaders, counts, max_options = build_loaders(args, processor, tokenizer)

    # 2. Model, optimizer and LR schedule (linear warmup, then cosine decay, stepped per update).
    #    Loss scaling is only needed for float16; bfloat16 runs without it.
    model, commit, lora_targets = build_model(args)
    base.sanity_check(model, train_loader, args.modality)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(params, lr=args.learning_rate, fused=base.DEVICE.type == "cuda")
    steps_per_epoch = min(len(train_loader), args.max_steps or len(train_loader))
    total_updates = math.ceil(steps_per_epoch / args.grad_accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=round(args.warmup_ratio * total_updates), num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=base.DEVICE.type == "cuda" and base.DTYPE == torch.float16)

    # 3. Baseline: JPT before any training. A fresh LoRA starts at zero, so this is plain JPT.
    history = []
    if not args.skip_baseline:
        print("Epoch 0 (before training):", flush=True)
        history.append({"epoch": 0, "validation": validate(model, val_loaders, args.modality)})

    # 4. Train, validating after every epoch and keeping the adapter with the best macro Brier.
    best_epoch, best_brier, best_state = None, math.inf, None
    run_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, params, args, epoch)
        print(f"Epoch {epoch} validation:", flush=True)
        validation = validate(model, val_loaders, args.modality)
        history.append({"epoch": epoch, "train": train_stats, "validation": validation})
        if validation["overall"]["macro_brier"] < best_brier:
            best_epoch, best_brier = epoch, validation["overall"]["macro_brier"]
            best_state = adapter_snapshot(model)
    elapsed = time.perf_counter() - run_started
    print(f"Training + validation: {elapsed / 60:.1f} minutes", flush=True)

    if best_epoch != args.epochs:
        set_peft_model_state_dict(model, best_state)
    print(f"Keeping epoch {best_epoch} (macro validation Brier {best_brier:.5f})", flush=True)
    baseline_brier = history[0]["validation"]["overall"]["macro_brier"] if not args.skip_baseline else None
    if baseline_brier is not None and best_brier >= baseline_brier:
        print(f"WARNING: no epoch beat untrained JPT (macro Brier {baseline_brier:.5f}); "
              "the saved adapter is worse than no adapter", flush=True)

    # 5. Everything needed to reproduce or compare this run.
    record = {
        "run_name": args.run_name,
        "model": args.model_id,
        "revision_requested": args.revision,
        "model_commit": commit,
        "modality": args.modality,
        "datasets": args.datasets,
        "derived_datasets": {name: DERIVED[name][0] for name in args.datasets if name in DERIVED},
        "sample_counts": counts,
        "max_options": max_options,
        "held_out": sorted(HELD_OUT),
        "learning_rate": args.learning_rate,
        "lr_schedule": "linear_warmup_cosine",
        "warmup_ratio": args.warmup_ratio,
        "objective": score_objective,
        "score_sigma": args.score_sigma if args.modality == "score" else None,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "selection_metric": "macro_validation_brier",
        "best_macro_brier": best_brier,
        "baseline_macro_brier": baseline_brier,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "max_grad_norm": base.MAX_GRAD_NORM,
        "lora": {**LORA_SETTINGS, "target_modules": lora_targets},
        "option_shuffling": "choice training rows" if args.modality == "choice" else "none",
        "ece_bins": base.ECE_BINS,
        "dtype": str(base.DTYPE),
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_steps_per_epoch": args.max_steps,
        "seed": args.seed,
        "packages": package_versions(),
        "weights_saved": not args.no_save,
        "elapsed_minutes": round(elapsed / 60, 2),
        "history": history,
    }
    save_run(args, record, model, processor, tokenizer)


if __name__ == "__main__":
    main()
