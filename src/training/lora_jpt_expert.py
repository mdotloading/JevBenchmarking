"""Train one JevBench modality expert across multiple datasets.

Examples (from the repository root):
    python -m src.training.lora_jpt_expert --modality score --batch-size 8
    python -m src.training.lora_jpt_expert --modality noul --batch-size 8
    python -m src.training.lora_jpt_expert --modality choice --datasets banking77 massive mnli

Defaults: up to 1,500 training and 150 validation rows per dataset. Samples
come only from their respective splits; test splits are never loaded.
Training uses the same hard-label Brier objective as lora_jpt_one_ds.
"""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import time

from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
from llm2jev.prompt import options_of

from src.training import lora_jpt_one_ds as base
from src.training.dataset import DATASET_ID, HF_TOKEN


# Config names and modalities from the JevBench dataset card.
# https://huggingface.co/datasets/Praveenrajus/jev-bench
DATASETS_BY_MODALITY = {
    "score": (
        "sst5", "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity",
        "stsb", "measuring_hate_speech",
    ),
    "noul": (
        "boolq", "fever_evidence", "paws", "civil_comments", "sms_spam",
        "strategyqa_closed", "strategyqa_grounded",
    ),
    "choice": (
        "banking77", "clinc150", "massive", "ledgar", "go_emotions",
        "mmlu", "arc_challenge", "mnli",
    ),
}
DEFAULT_DATASETS = {
    "score": ("sst5", "yelp5", "helpsteer2_helpfulness", "stsb"),
    "noul": ("boolq", "fever_evidence", "paws", "civil_comments"),
    "choice": ("banking77", "massive", "go_emotions", "mnli"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=DATASETS_BY_MODALITY, default="score")
    parser.add_argument("--datasets", nargs="+", help="JevBench configs for the selected modality; default: four")
    parser.add_argument("--train-samples", type=int, default=1500, help="Maximum rows per training split")
    parser.add_argument("--validation-samples", type=int, default=150, help="Maximum rows per validation split")
    parser.add_argument("--model-id", default=base.MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--validation-batch-size", type=int, default=base.VALIDATION_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=base.NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=importlib.util.find_spec("fla") is None,
        help="Trade speed for VRAM; enabled by default when FLA is absent",
    )
    parser.add_argument("--print-every-batches", type=int, default=base.PRINT_EVERY_BATCHES)
    parser.add_argument("--max-steps", type=int, help="Maximum microbatches per epoch for a short check")
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoint weights; still record run metadata")
    args = parser.parse_args()
    for name in ("train_samples", "validation_samples", "batch_size", "validation_batch_size",
                 "grad_accum_steps", "epochs", "print_every_batches"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    args.datasets = args.datasets or list(DEFAULT_DATASETS[args.modality])
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    invalid = set(args.datasets) - set(DATASETS_BY_MODALITY[args.modality])
    if invalid:
        parser.error(f"Not trainable {args.modality} configs: {sorted(invalid)}. "
                     f"Choose from {', '.join(DATASETS_BY_MODALITY[args.modality])}")
    return args


def load_samples(name, split, limit, seed):
    dataset = load_dataset(DATASET_ID, name, split=split, token=HF_TOKEN)
    if not len(dataset):
        raise ValueError(f"{name}/{split} is empty")
    sampled = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    if len(sampled) < limit:
        print(f"{name}/{split}: using all {len(sampled)} available rows (requested {limit})", flush=True)
    return sampled


def prepare_loaders(args, processor, tokenizer):
    raw_train, raw_validation, counts = {}, {}, {}
    for index, name in enumerate(args.datasets):
        raw_train[name] = load_samples(name, "train", args.train_samples, args.seed + index)
        raw_validation[name] = load_samples(name, "validation", args.validation_samples, args.seed + index)
        counts[name] = {"train": len(raw_train[name]), "validation": len(raw_validation[name])}
        print(f"{name}: {counts[name]}", flush=True)
    num_options = max(
        len(options_of(json.loads(question))[0])
        for dataset in (*raw_train.values(), *raw_validation.values())
        for question in dataset["question"]
    )
    labels, token_ids = base.build_llm2jev_labels(processor, tokenizer, num_options)
    collator = base.JevBenchCollator(tokenizer, token_ids)
    train_parts = [
        base.prepare_dataset(dataset, processor, tokenizer, labels, args.modality)
        for dataset in raw_train.values()
    ]
    mixed_train = concatenate_datasets(train_parts)
    pin_memory = base.DEVICE.type == "cuda"
    train_loader = DataLoader(mixed_train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, pin_memory=pin_memory)
    validation_loaders = {
        name: DataLoader(
            base.prepare_dataset(dataset, processor, tokenizer, labels, args.modality),
            batch_size=args.validation_batch_size, collate_fn=collator, pin_memory=pin_memory,
        )
        for name, dataset in raw_validation.items()
    }
    return train_loader, validation_loaders, counts


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if base.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name(), flush=True)
    print(f"Modality={args.modality}, datasets={args.datasets}, device={base.DEVICE}, dtype={base.DTYPE}, "
          f"microbatch={args.batch_size}, effective_batch={args.batch_size * args.grad_accum_steps}, "
          f"gradient_checkpointing={args.gradient_checkpointing}", flush=True)
    for package in ("fla", "causal_conv1d"):
        print(f"Optional kernel package {package}: {importlib.util.find_spec(package) is not None}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train_loader, validation_loaders, counts = prepare_loaders(args, processor, tokenizer)

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, dtype=base.DTYPE, attn_implementation="sdpa",
    ).to(base.DEVICE)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=base.find_lora_target_modules(model),
    ))
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.sanity_check(model, train_loader, args.modality)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=base.LEARNING_RATE, fused=base.DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=base.DEVICE.type == "cuda" and base.DTYPE == torch.float16)
    validation_history = []
    run_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = min(len(train_loader), args.max_steps or len(train_loader))
        total_loss = torch.zeros((), device=base.DEVICE)
        total_correct = torch.zeros((), device=base.DEVICE, dtype=torch.long)
        seen = tokens = window_examples = updates = 0
        recent_losses = deque(maxlen=base.ROLLING_WINDOW)
        recent_correct = deque(maxlen=base.ROLLING_WINDOW)
        if base.DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        print(f"Epoch {epoch+1}: training {len(train_loader.dataset)} mixed examples, {steps} batches", flush=True)
        for step, batch in enumerate(train_loader):
            if step >= steps:
                break
            if step == 0:
                print(f"First batch shape: {tuple(batch['input_ids'].shape)}; starting forward/backward", flush=True)
            result = base.forward_batch(model, batch, args.modality)
            n = batch["input_ids"].shape[0]
            scaler.scale(result["loss"] * n).backward()
            window_examples += n
            boundary = (step + 1) % args.grad_accum_steps == 0 or step + 1 == steps
            if boundary:
                scaler.unscale_(optimizer)
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(window_examples)
                torch.nn.utils.clip_grad_norm_(parameters, base.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                window_examples = 0
                updates += 1
            total_loss += result["per_example_loss"].sum()
            total_correct += result["correct_mask"].sum()
            seen += n
            tokens += int(batch["attention_mask"].sum())
            recent_losses.append(result["per_example_loss"])
            recent_correct.append(result["correct_mask"])
            if step == 0 or (step + 1) % args.print_every_batches == 0 or step + 1 == steps:
                if base.DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                rolling_loss = torch.cat(list(recent_losses))[-base.ROLLING_WINDOW:].mean().item()
                rolling_acc = torch.cat(list(recent_correct))[-base.ROLLING_WINDOW:].float().mean().item()
                remaining = (steps - step - 1) * elapsed / (step + 1)
                print(f"epoch={epoch+1} batch={step+1}/{steps} updates={updates} "
                      f"loss={total_loss.item()/seen:.6f} acc={total_correct.item()/seen:.4f} "
                      f"rolling_loss={rolling_loss:.6f} rolling_acc={rolling_acc:.4f} "
                      f"examples/s={seen/elapsed:.2f} tokens/s={tokens/elapsed:.0f} "
                      f"train_ETA_min={remaining/60:.1f}", flush=True)
        if base.DEVICE.type == "cuda":
            print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        per_dataset = {}
        for name, loader in validation_loaders.items():
            print(f"Validating {name}: {len(loader.dataset)} examples", flush=True)
            per_dataset[name] = base.evaluate(model, loader, args.modality)
            print(f"{name}: {per_dataset[name]}", flush=True)
        total_n = sum(result["n"] for result in per_dataset.values())
        overall = {
            "loss": sum(result["loss"] * result["n"] for result in per_dataset.values()) / total_n,
            "accuracy": sum(result["correct"] for result in per_dataset.values()) / total_n,
            "macro_accuracy": sum(result["accuracy"] for result in per_dataset.values()) / len(per_dataset),
            "n": total_n,
        }
        validation_history.append({"epoch": epoch + 1, "datasets": per_dataset, "overall": overall})
        print(f"Combined validation: {overall}", flush=True)

    run_seconds = time.perf_counter() - run_started
    metadata = {
        "model": args.model_id, "modality": args.modality, "datasets": args.datasets,
        "sample_counts": counts, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_steps_per_epoch": args.max_steps, "checkpoints_saved": not args.no_save,
        "elapsed_seconds": round(run_seconds, 3), "elapsed_minutes": round(run_seconds / 60, 2),
        "scope": "training_and_validation", "approximate": False,
        "validation_history": validation_history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"jpt-{args.modality}-expert"
    metadata_path = args.output_dir / f"{stem}-run.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "lora_run_times.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({key: value for key, value in metadata.items() if key != "validation_history"}) + "\n")
    print(f"LoRA training + validation: {run_seconds / 60:.2f} minutes; metadata: {metadata_path}", flush=True)
    if not args.no_save:
        lora_dir = args.output_dir / f"{stem}-lora"
        merged_dir = args.output_dir / f"{stem}-merged"
        base.save_lora(model, processor, tokenizer, lora_dir)
        base.merge_and_save(model, processor, tokenizer, merged_dir)
        for directory in (lora_dir, merged_dir):
            (directory / "training_metadata.json").write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
