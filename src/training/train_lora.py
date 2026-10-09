"""LoRA-tune JPT or Decision 2.0 on one dataset (specialist) or one question type (expert).

    python -m src.training.train_lora                                   # uses the settings below
    python -m src.training.train_lora --model jpt --train-on choice
    python -m src.training.train_lora --train-on boolq --max-steps 20 --no-save   # smoke test

Saves checkpoints/<model>-<dataset>-lora or checkpoints/<model>-<modality>-expert-lora; the benchmark finds it there.
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule, get_cosine_schedule_with_warmup

from src.data.jevbench import DATASETS_BY_MODALITY, EXPERT_MIX, HELD_OUT, MODALITY_OF, load_examples, source_of
from src.metrics import brier, expected_calibration_error, predicted_index, training_objective
from src.models import load_model

# ============================================================
# SETTINGS
# ============================================================
MODEL = "decision2"      # "jpt" | "decision2"
TRAIN_ON = "banking77"   # a dataset -> specialist | "choice" / "noul" / "score" -> expert on EXPERT_MIX
PROFILES = {
    "specialist": dict(lora_rank=16, lora_alpha=16, learning_rate=5e-5, epochs=1, lr_schedule="constant",
                       warmup_ratio=0.0, train_rows_per_dataset=None, validation_rows=500, score_sigma=0.0),
    "expert": dict(lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=1, lr_schedule="cosine",
                   warmup_ratio=0.05, train_rows_per_dataset=2500, validation_rows=500, score_sigma=1.0),
}
LORA_DROPOUT = 0.05
BATCH_SIZE = 4           # 8 GB GPU: lower to 2 (and raise GRAD_ACCUM_STEPS) on CUDA out of memory
GRAD_ACCUM_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
PRINT_EVERY_BATCHES = 25
CHECKPOINTS = Path("checkpoints")


def training_target(train_on):
    """-> (kind, modality, datasets) for a dataset or modality name; refuses held-out datasets."""
    if train_on in DATASETS_BY_MODALITY:
        kind, modality, datasets = "expert", train_on, list(EXPERT_MIX[train_on])
    elif train_on in MODALITY_OF:
        kind, modality, datasets = "specialist", MODALITY_OF[train_on], [train_on]
    else:
        raise SystemExit(f"--train-on must be a modality {list(DATASETS_BY_MODALITY)} or a jev-bench dataset")
    held_out = [name for name in datasets if source_of(name) in HELD_OUT]
    if held_out:
        raise SystemExit(f"{held_out} are held out for zero-shot testing and can never be trained on")
    return kind, modality, datasets


def build_loaders(model, datasets, profile, batch_size):
    """One shuffled training loader over all datasets (choice options shuffled), one validation loader per dataset."""
    train_items, validation_loaders, row_counts = [], {}, {}
    for i, name in enumerate(datasets):
        train = load_examples(name, "train", profile["train_rows_per_dataset"], seed=SEED + i)
        validation = load_examples(name, "validation", profile["validation_rows"], seed=SEED + i)
        train_items += [model.encode(example, shuffle_seed=SEED) for example in train]
        validation_loaders[name] = DataLoader([model.encode(example) for example in validation],
                                              batch_size=batch_size, collate_fn=model.collate)
        row_counts[name] = {"train": len(train), "validation": len(validation)}
        print(f"{name}: {row_counts[name]}", flush=True)
    train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=model.collate,
                              generator=torch.Generator().manual_seed(SEED))
    return train_loader, validation_loaders, row_counts


def validate(model, loaders, modality):
    """Brier, accuracy and ECE per dataset, plus their macro averages."""
    model.network.eval()
    results = {}
    with torch.inference_mode():
        for name, loader in loaders.items():
            briers, corrects, confidences = [], [], []
            for batch in loader:
                logits, _, gold = model.option_logits(batch)
                probabilities = torch.softmax(logits, dim=-1)
                predicted = predicted_index(probabilities, modality)
                briers.append(brier(probabilities, gold).cpu())
                corrects.append((predicted == gold).cpu())
                confidences.append(probabilities.gather(1, predicted[:, None]).squeeze(1).cpu())
            briers, corrects, confidences = torch.cat(briers), torch.cat(corrects), torch.cat(confidences)
            results[name] = {"brier": briers.mean().item(), "accuracy": corrects.float().mean().item(),
                             "ece": expected_calibration_error(confidences, corrects), "n": len(briers)}
            print(f"  {name}: {results[name]}", flush=True)
    model.network.train()
    macro = {f"macro_{key}": sum(r[key] for r in results.values()) / len(results) for key in ("brier", "accuracy", "ece")}
    print(f"  macro: {macro}", flush=True)
    return {"datasets": results, **macro}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, choices=("jpt", "decision2"))
    p.add_argument("--train-on", default=TRAIN_ON)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--max-steps", type=int, help="Cap batches per epoch (smoke test)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=importlib.util.find_spec("fla") is None, help="Default: on when the fla kernels are missing")
    args = p.parse_args()

    kind, modality, datasets = training_target(args.train_on)
    profile = PROFILES[kind]
    run_name = f"{args.model}-{args.train_on}{'-expert' if kind == 'expert' else ''}-lora"
    output = CHECKPOINTS / run_name
    if output.exists() and not args.no_save:
        raise SystemExit(f"{output} already exists; move it away first")
    torch.manual_seed(SEED)
    print(f"{run_name}: {kind} on {datasets} ({modality}), {profile}, batch {args.batch_size}x{args.grad_accum_steps}",
          flush=True)

    # 1. Model and data
    model = load_model(args.model)
    train_loader, validation_loaders, row_counts = build_loaders(model, datasets, profile, args.batch_size)
    lora_targets = model.add_lora(profile["lora_rank"], profile["lora_alpha"], LORA_DROPOUT)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.network.train()
    trainable = [p for p in model.network.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}", flush=True)

    steps_per_epoch = min(len(train_loader), args.max_steps or len(train_loader))
    total_updates = math.ceil(steps_per_epoch / args.grad_accum_steps) * profile["epochs"]
    optimizer = torch.optim.AdamW(trainable, lr=profile["learning_rate"])
    scheduler = (get_constant_schedule(optimizer) if profile["lr_schedule"] == "constant" else
                 get_cosine_schedule_with_warmup(optimizer, round(profile["warmup_ratio"] * total_updates), total_updates))

    # 2. Baseline (a fresh LoRA is zero, so this is the untuned model), then train and keep the best epoch.
    print("Epoch 0 (before training):", flush=True)
    history = [{"epoch": 0, "validation": validate(model, validation_loaders, modality)}]
    best_epoch, best_brier, best_weights = 0, history[0]["validation"]["macro_brier"], None
    started = time.perf_counter()
    for epoch in range(1, profile["epochs"] + 1):
        loss_sum, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            logits, option_mask, gold = model.option_logits(batch)
            loss = training_objective(logits, gold, option_mask, modality, profile["score_sigma"])
            (loss.mean() / args.grad_accum_steps).backward()
            if (step + 1) % args.grad_accum_steps == 0 or step + 1 == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum, seen = loss_sum + loss.detach().sum().item(), seen + len(loss)
            if step == 0 or (step + 1) % PRINT_EVERY_BATCHES == 0 or step + 1 == steps_per_epoch:
                print(f"epoch {epoch} batch {step + 1}/{steps_per_epoch} loss {loss_sum / seen:.4f} "
                      f"{seen / (time.perf_counter() - started):.1f} examples/s", flush=True)
        print(f"Epoch {epoch}:", flush=True)
        validation = validate(model, validation_loaders, modality)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "validation": validation})
        if validation["macro_brier"] < best_brier:
            best_epoch, best_brier = epoch, validation["macro_brier"]
            best_weights = {name: p.detach().cpu().clone()
                            for name, p in model.network.named_parameters() if p.requires_grad}
    if best_weights is None:
        print("WARNING: no epoch beat the untuned model; keeping the last epoch", flush=True)
        best_epoch = profile["epochs"]
    elif best_epoch != profile["epochs"]:
        with torch.no_grad():
            for name, p in model.network.named_parameters():
                if name in best_weights:
                    p.copy_(best_weights[name])
    print(f"Keeping epoch {best_epoch} (macro validation Brier {best_brier:.4f}), "
          f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)

    # 3. Save adapter + run record.
    if args.no_save:
        return
    output.mkdir(parents=True)
    model.save_adapter(output)
    record = {"model": args.model, "train_on": args.train_on, "kind": kind, "modality": modality, "datasets": datasets,
              "row_counts": row_counts, "profile": profile, "lora_dropout": LORA_DROPOUT, "lora_targets": lora_targets,
              "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps, "seed": SEED,
              "best_epoch": best_epoch, "history": history}
    (output / "training_metadata.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
