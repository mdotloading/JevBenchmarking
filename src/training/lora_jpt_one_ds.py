

"""Shared building blocks for LoRA-tuning JPT, plus a single-dataset trainer.
 
This file is used two ways:
  * as a library: lora_jpt_expert.py imports the helpers below (rendering,
    collating, the forward pass, evaluation, saving);
  * as a script: tune JPT on ONE of banking77, boolq or helpsteer2_helpfulness.
        python -m src.training.lora_jpt_one_ds --dataset banking77
 
Training objective: multi-class Brier score over the option labels, read from
the next-token logits at the answer position of llm2jev's chat prompt.
"""
from collections import deque
import json
from pathlib import Path
import random
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
from llm2jev.prompt import render, find_labels, options_of
 
from src.data.parser import parse_row
from src.training.dataset import (
    load_banking77_train,
    load_banking77_validation,
    load_helpsteer_train,
    load_helpsteer_validation,
    load_boolq_train,
    load_boolq_validation,
)
 
 
# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "kirp/jpt-0.8b"
 
DATASETS = {
    "banking77": {"train": load_banking77_train, "validation": load_banking77_validation, "primitive": "choice"},
    "boolq": {"train": load_boolq_train, "validation": load_boolq_validation, "primitive": "noul"},
    "helpsteer2_helpfulness": {"train": load_helpsteer_train, "validation": load_helpsteer_validation,
                               "primitive": "score"},
}
 
SEED = 42
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5
 
# RTX 4060 (8 GB): start at 4, try 8 if VRAM is comfortable, drop to 2 on CUDA OOM.
BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 4
VALIDATION_SAMPLES = 500
 
PRINT_EVERY_BATCHES = 25
ROLLING_WINDOW = 100
MAX_GRAD_NORM = 1.0
ECE_BINS = 15  # equal-width confidence bins for expected calibration error
 
 
# ============================================================
# DEVICE, DTYPE, SEED
# ============================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
 
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
 
 
# ============================================================
# LABELS AND LORA TARGETS
# ============================================================
def build_llm2jev_labels(processor, tokenizer, num_options):
    """Find the token ids of llm2jev's option labels (A, B, ..., Z, AA, AB, ...).
 
    Tokenization depends on context (" A" after "Answer:" is not the same as "A"
    on its own), so the labels are looked up at the end of a real rendered prompt.
    """
    _, probe_prompts, _ = render(
        processor=processor, state="x", questions={"q": {"type": "noul"}},
        labels=["A", "B"], style="chat",
    )
    probe_prompt, _ = probe_prompts["q"]
    return find_labels(tokenizer, probe_prompt, n=num_options)
 
 
# Linear layers that get a LoRA adapter: full attention, MLP and Qwen3.5's DeltaNet.
LORA_LEAF_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",                               # attention
    "gate_proj", "up_proj", "down_proj",                                  # MLP
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",    # DeltaNet
}
 
 
def find_lora_target_modules(model):
    """Full names of the language model's linear layers to adapt. Vision tower and lm_head are skipped."""
    targets = []
    for name, module in model.named_modules():
        lower = name.lower()
        if not isinstance(module, nn.Linear):
            continue
        if "vision" in lower or "visual" in lower or "lm_head" in lower:
            continue
        if name.split(".")[-1] in LORA_LEAF_NAMES:
            targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules found.")
    return targets
 
 
# ============================================================
# DATA: RENDER, TOKENIZE, COLLATE
# ============================================================
def shuffle_choice_options(question, row_id, seed):
    """A copy of a choice question with its options in a random but reproducible order.
 
    Without this, every banking77 row lists the 77 intents in the same order, and
    the model can learn "the answer is usually near position X" instead of reading
    the options. The order depends only on (seed, row id), so reruns match.
    """
    items = list(question["criteria"].items())
    random.Random(f"{seed}:{row_id}").shuffle(items)
    return {**question, "criteria": dict(items)}
 
 
def prepare_dataset(dataset, processor, tokenizer, labels, primitive, shuffle_options=False, seed=SEED):
    """Render every row with llm2jev's prompt and tokenize it once, before the GPU loop.
 
    shuffle_options: shuffle the option order of choice questions (use for training
    data only). Score levels are never shuffled because their order is their meaning,
    and noul questions always have the same two options.
    """
    def encode(rows):
        prompts, targets, option_counts = [], [], []
        for i in range(len(rows["state"])):
            example = parse_row({key: values[i] for key, values in rows.items()})
            if example["primitive"] != primitive or example["question"]["type"] != primitive:
                raise ValueError(f"Expected {primitive!r} examples, got {example['id']!r}")
 
            question = example["question"]
            if shuffle_options and primitive == "choice" and isinstance(question.get("criteria"), dict):
                question = shuffle_choice_options(question, example["id"], seed)
 
            _, rendered, media = render(
                processor=processor, state=example["state"],
                questions={"benchmark_test": question}, labels=labels, style="chat",
            )
            if media:
                raise ValueError("This training script supports text-only examples")
            prompt, keys = rendered["benchmark_test"]
 
            # Map the gold label to the option's position in the rendered prompt.
            target = example["label"]
            if primitive == "noul":
                target = {0: "false", 1: "true"}[target]
            elif primitive == "score":
                target = str(target)
            prompts.append(prompt)
            targets.append(keys.index(target))
            option_counts.append(len(keys))
 
        encoded = tokenizer(prompts, add_special_tokens=False, return_attention_mask=False)
        return {"input_ids": encoded["input_ids"], "target_indices": targets, "num_options": option_counts}
 
    encoded = dataset.map(
        encode, batched=True, batch_size=256, remove_columns=dataset.column_names,
        load_from_cache_file=False,  # a prompt or template change must never reuse stale tokens
        desc=f"Render/tokenize {primitive}",
    )
    lengths = [len(ids) for ids in encoded["input_ids"]]
    print(f"Prompt tokens: mean={sum(lengths) / len(lengths):.0f}, max={max(lengths)}")
    return encoded
 
 
class JevBenchCollator:
    """Pads a list of tokenized rows into one batch and marks which option slots exist."""
 
    def __init__(self, tokenizer, option_token_ids):
        self.tokenizer = tokenizer
        self.option_token_ids = torch.tensor(option_token_ids, dtype=torch.long)
 
    def __call__(self, rows):
        encoded = self.tokenizer.pad(
            {"input_ids": [row["input_ids"] for row in rows]},
            padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        num_options = torch.tensor([row["num_options"] for row in rows])
        return {
            **encoded,
            "target_indices": torch.tensor([row["target_indices"] for row in rows]),
            "option_token_ids": self.option_token_ids,
            # Rows in a batch can have different option counts; mask the slots a row doesn't have.
            "option_mask": torch.arange(len(self.option_token_ids))[None, :] < num_options[:, None],
        }
 
 
# ============================================================
# FORWARD PASS
# ============================================================
def decision_logits(model, input_ids, attention_mask, option_token_ids):
    """Logits of the option-label tokens at each row's last prompt position.
 
    Only the label rows of the output head are used, instead of computing logits
    for the whole vocabulary at every position, which saves a lot of memory.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.config.model_type != "qwen3_5":
        raise ValueError("Selective decision projection requires the Qwen3.5 JPT architecture")
    hidden = base.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
    ).last_hidden_state
    last = attention_mask.sum(dim=1) - 1  # last real token (right padding)
    hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    head = base.get_output_embeddings()
    weight = head.weight.index_select(0, option_token_ids)
    bias = head.bias.index_select(0, option_token_ids) if head.bias is not None else None
    return F.linear(hidden, weight, bias).float()
 
 
def forward_batch(model, batch, primitive="choice"):
    """Option probabilities, Brier loss and predictions for one batch."""
    tensors = {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}
    with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
        option_logits = decision_logits(
            model, tensors["input_ids"], tensors["attention_mask"], tensors["option_token_ids"],
        )
    option_logits = option_logits.masked_fill(~tensors["option_mask"], float("-inf"))
    probabilities = F.softmax(option_logits, dim=-1)
 
    # Multi-class Brier score: squared distance between the predicted distribution and the one-hot answer.
    targets = F.one_hot(tensors["target_indices"], num_classes=probabilities.shape[-1]).float()
    per_example_loss = (probabilities - targets).pow(2).sum(dim=-1)
 
    predictions = probabilities.argmax(dim=-1)
    if primitive == "score":
        # Score answers are the rounded expected level, matching llm2jev and the benchmark.
        levels = torch.arange(probabilities.shape[-1], device=probabilities.device)
        predictions = (probabilities * levels).sum(dim=-1).round().long()
 
    return {
        "loss": per_example_loss.mean(),
        "per_example_loss": per_example_loss.detach(),
        "prediction_indices": predictions.detach(),
        "target_indices": tensors["target_indices"].detach(),
        "correct_mask": (predictions == tensors["target_indices"]).detach(),
        "probabilities": probabilities.detach(),
    }
 
 
# ============================================================
# EVALUATION
# ============================================================
def expected_calibration_error(confidence, correct, n_bins=ECE_BINS):
    """Top-label ECE: the gap between confidence and accuracy, averaged over
    equal-width confidence bins and weighted by how many rows fall in each bin.
    0 means perfectly calibrated."""
    bins = torch.clamp((confidence * n_bins).long(), max=n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            gap = (confidence[in_bin].mean() - correct[in_bin].float().mean()).abs().item()
            ece += in_bin.float().mean().item() * gap
    return ece
 
 
def evaluate(model, dataloader, primitive="choice", n_bins=ECE_BINS):
    """Brier loss, accuracy and ECE over a whole dataset.
 
    Confidence is the probability the model gave to the option it predicted.
    For score questions that is the probability of the rounded expected level.
    """
    model.eval()
    losses, corrects, confidences = [], [], []
    with torch.inference_mode():
        for batch in dataloader:
            result = forward_batch(model, batch, primitive)
            chosen = result["prediction_indices"][:, None]
            losses.append(result["per_example_loss"].cpu())
            corrects.append(result["correct_mask"].cpu())
            confidences.append(result["probabilities"].gather(1, chosen).squeeze(1).cpu())
    model.train()
 
    losses, corrects, confidences = torch.cat(losses), torch.cat(corrects), torch.cat(confidences)
    n = len(losses)
    brier = losses.sum().item() / n
    correct = int(corrects.sum())
    return {
        "loss": brier,     # kept under its old name so existing scripts still work
        "brier": brier,
        "accuracy": correct / n,
        "ece": expected_calibration_error(confidences, corrects, n_bins),
        "correct": correct,
        "n": n,
    }
 
 
def sanity_check(model, dataloader, primitive="choice"):
    """Run one batch and check the shapes and that every probability row sums to 1."""
    batch = next(iter(dataloader))
    print("\n=== Batch sanity check ===")
    for key in ("input_ids", "attention_mask", "target_indices", "option_token_ids"):
        print(f"{key}: {tuple(batch[key].shape)}")
 
    model.eval()
    with torch.inference_mode():
        result = forward_batch(model, batch, primitive)
    model.train()
 
    sums = result["probabilities"].sum(dim=-1)
    print(f"probabilities: {tuple(result['probabilities'].shape)}, batch loss: {result['loss'].item():.4f}")
    assert result["probabilities"].shape[1] == batch["option_token_ids"].numel()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), f"probability sums: {sums}"
    print("Sanity check passed.")
 
 
# ============================================================
# SAVING
# ============================================================
def save_lora(model, processor, tokenizer, output_dir):
    """Save only the LoRA adapter (small) plus the processor and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved LoRA adapter to {output_dir}")
 
 
def merge_and_save(model, processor, tokenizer, output_dir):
    """Fold the adapter into the base weights and save a standalone model for serving.
 
    Call this last: merge_and_unload changes the model in place.
    """
    print("\nMerging LoRA...")
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")
 
 
# ============================================================
# MAIN: TUNE ON ONE DATASET
# ============================================================
def main():
    import argparse
    import importlib.util
    import time
 
    parser = argparse.ArgumentParser(description="Train JPT on one jev-bench dataset with LoRA and Brier loss")
    parser.add_argument("--dataset", choices=DATASETS, default="banking77")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Physical GPU micro-batch")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=importlib.util.find_spec("fla") is None,
                        help="Trade speed for VRAM (default: on when the fla kernels are missing)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument("--max-steps", type=int, help="Limit micro-batches per epoch, for speed checks")
    parser.add_argument("--no-save", action="store_true", help="Skip checkpoint writes")
    args = parser.parse_args()
    if min(args.batch_size, args.grad_accum_steps, args.epochs, args.validation_samples) < 1:
        parser.error("Batch size, accumulation steps, epochs and validation samples must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
 
    config = DATASETS[args.dataset]
    primitive = config["primitive"]
    print(f"Dataset={args.dataset}, primitive={primitive}, device={DEVICE}, dtype={DTYPE}, "
          f"batch={args.batch_size}x{args.grad_accum_steps}, gradient_checkpointing={args.gradient_checkpointing}")
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name())
 
    # --- Data ---
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train_raw = config["train"]()
    val_raw = config["validation"]()
    val_raw = val_raw.select(range(min(args.validation_samples, len(val_raw))))
    if not len(train_raw) or not len(val_raw):
        raise ValueError("Training and validation datasets must be nonempty")
 
    num_options = max(
        len(options_of(json.loads(q))[0]) for ds in (train_raw, val_raw) for q in ds["question"]
    )
    labels, token_ids = build_llm2jev_labels(processor, tokenizer, num_options)
    train_data = prepare_dataset(train_raw, processor, tokenizer, labels, primitive, shuffle_options=True)
    val_data = prepare_dataset(val_raw, processor, tokenizer, labels, primitive)
    collator = JevBenchCollator(tokenizer, token_ids)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=VALIDATION_BATCH_SIZE, collate_fn=collator, pin_memory=pin)
 
    # --- Model ---
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=find_lora_target_modules(model),
    ))
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sanity_check(model, train_loader, primitive)
 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LEARNING_RATE, fused=DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and DTYPE == torch.float16)
 
    # --- Train ---
    run_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = min(len(train_loader), args.max_steps or len(train_loader))
        loss_sum = torch.zeros((), device=DEVICE)
        correct_sum = torch.zeros((), device=DEVICE, dtype=torch.long)
        seen = tokens = examples_in_window = updates = 0
        recent_loss, recent_correct = deque(maxlen=ROLLING_WINDOW), deque(maxlen=ROLLING_WINDOW)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
 
        for step, batch in enumerate(train_loader):
            if step >= steps:
                break
            result = forward_batch(model, batch, primitive)
            n = batch["input_ids"].shape[0]
 
            # Sum example losses, then divide gradients once per accumulation window:
            # the exact per-example mean, even for a short final window.
            scaler.scale(result["loss"] * n).backward()
            examples_in_window += n
            if (step + 1) % args.grad_accum_steps == 0 or step + 1 == steps:
                scaler.unscale_(optimizer)
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(examples_in_window)
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                examples_in_window = 0
                updates += 1
 
            # Metrics stay on the GPU; they are copied to the CPU only when printing.
            loss_sum += result["per_example_loss"].sum()
            correct_sum += result["correct_mask"].sum()
            recent_loss.append(result["per_example_loss"])
            recent_correct.append(result["correct_mask"])
            seen += n
            tokens += int(batch["attention_mask"].sum())
            if (step + 1) % PRINT_EVERY_BATCHES == 0 or step + 1 == steps:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                rolling_loss = torch.cat(list(recent_loss))[-ROLLING_WINDOW:].mean().item()
                rolling_acc = torch.cat(list(recent_correct))[-ROLLING_WINDOW:].float().mean().item()
                eta_min = (steps - step - 1) * elapsed / (step + 1) / 60
                print(f"epoch={epoch} batch={step + 1}/{steps} updates={updates} "
                      f"loss={loss_sum.item() / seen:.5f} acc={correct_sum.item() / seen:.4f} "
                      f"rolling_loss={rolling_loss:.5f} rolling_acc={rolling_acc:.4f} "
                      f"examples/s={seen / elapsed:.1f} tokens/s={tokens / elapsed:.0f} ETA_min={eta_min:.1f}")
 
        if DEVICE.type == "cuda":
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        validation = evaluate(model, val_loader, primitive)
        print(f"Validation: brier={validation['brier']:.5f}, accuracy={validation['accuracy']:.4f}, "
              f"ece={validation['ece']:.4f}")
 
    # --- Log and save ---
    run_seconds = time.perf_counter() - run_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lora_run_times.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "dataset": args.dataset, "model": MODEL_ID, "learning_rate": LEARNING_RATE,
            "batch_size": args.batch_size, "epochs": args.epochs, "grad_accum_steps": args.grad_accum_steps,
            "max_steps_per_epoch": args.max_steps, "gradient_checkpointing": args.gradient_checkpointing,
            "option_shuffling": "choice training rows",
            "elapsed_minutes": round(run_seconds / 60, 2), "final_validation": validation,
        }) + "\n")
    print(f"LoRA training + validation: {run_seconds / 60:.2f} minutes")
    if not args.no_save:
        save_lora(model, processor, tokenizer, args.output_dir / f"jpt-{args.dataset}-lora")
        merge_and_save(model, processor, tokenizer, args.output_dir / f"jpt-{args.dataset}-merged")
 
 
if __name__ == "__main__":
    main()
 
