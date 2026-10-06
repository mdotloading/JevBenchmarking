from collections import deque
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
)

from llm2jev.prompt import render, find_labels, options_of

from src.data.parser import parse_row
from src.training.dataset import (
    load_banking77_train,
    load_banking77_validation,
    load_helpsteer_train,
    load_helpsteer_validation,
    load_boolq_train,
    load_boolq_validation
)



# THIS SCRIPT IS ONLY USED FOR TUNING A JPT BACKBONE ON EITHER BANKING77, HELPSTEER OR BOOLQ... refer to a different script for tuning across different datasets in the same modality or all 3 modalities at once





# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "kirp/jpt-0.8b"

DATASETS = {
    "banking77": {
        "train": load_banking77_train,
        "validation": load_banking77_validation,
        "primitive": "choice",
    },
    "boolq": {
        "train": load_boolq_train,
        "validation": load_boolq_validation,
        "primitive": "noul",
    },
    "helpsteer2_helpfulness": {
        "train": load_helpsteer_train,
        "validation": load_helpsteer_validation,
        "primitive": "score",
    },
}

SEED = 42

NUM_EPOCHS = 1

LEARNING_RATE = 5e-5

# ------------------------------------------------------------
# START HERE ON YOUR RTX 4060.
#
# Try:
#     4
#
# If VRAM is comfortable:
#     8
#
# If CUDA OOM:
#     2
# ------------------------------------------------------------

BATCH_SIZE = 4

VALIDATION_BATCH_SIZE = 4

VALIDATION_SAMPLES = 500

PRINT_EVERY_BATCHES = 25

ROLLING_WINDOW = 100

MAX_GRAD_NORM = 1.0


# ============================================================
# DEVICE / DTYPE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    if torch.cuda.is_bf16_supported():
        DTYPE = torch.bfloat16
    else:
        DTYPE = torch.float16

else:

    DEVICE = torch.device("cpu")
    DTYPE = torch.float32


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LLM2JEV LABEL DISCOVERY
# ============================================================

def build_llm2jev_labels(
    processor,
    tokenizer,
    num_options,
):
    """
    Determine the exact tokenizer IDs used for llm2jev's
    decision labels:

        A
        B
        C
        ...
        Z
        AA
        AB
        ...

    This uses the REAL prompt ending because tokenization
    depends on context.

    We therefore do NOT just call:

        tokenizer.encode("A")

    in isolation.
    """

    _, probe_prompts, _ = render(
        processor=processor,
        state="x",
        questions={
            "q": {
                "type": "noul",
            }
        },
        labels=["A", "B"],
        style="chat",
    )

    probe_prompt, _ = probe_prompts["q"]

    labels, token_ids = find_labels(
        tokenizer,
        probe_prompt,
        n=num_options,
    )

    return labels, token_ids


# ============================================================
# FIND LORA TARGET MODULES
# ============================================================

def find_lora_target_modules(model):
    """
    Find language-model linear projections where LoRA
    adapters should be inserted.

    Full-attention projections:
        q_proj
        k_proj
        v_proj
        o_proj

    MLP projections:
        gate_proj
        up_proj
        down_proj

    Qwen3.5 DeltaNet projections:
        in_proj_qkv
        in_proj_z
        in_proj_b
        in_proj_a
        out_proj

    Vision modules and lm_head are excluded.
    """

    target_leaf_names = {
        # Full attention
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",

        # MLP
        "gate_proj",
        "up_proj",
        "down_proj",

        # DeltaNet
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    }

    target_modules = []

    for name, module in model.named_modules():

        if not isinstance(module, nn.Linear):
            continue

        lower_name = name.lower()

        if "vision" in lower_name:
            continue

        if "visual" in lower_name:
            continue

        if "lm_head" in lower_name:
            continue

        leaf_name = name.split(".")[-1]

        if leaf_name in target_leaf_names:
            target_modules.append(name)

    if not target_modules:

        raise RuntimeError(
            "No LoRA target modules found."
        )

    return target_modules


# ============================================================
# COLLATOR
# ============================================================

def prepare_dataset(dataset, processor, tokenizer, labels, primitive):
    """Render once and batch-tokenize before the GPU training loop."""
    def encode(rows):
        prompts, targets, option_counts = [], [], []
        for i in range(len(rows["state"])):
            example = parse_row({key: values[i] for key, values in rows.items()})
            if example["primitive"] != primitive or example["question"]["type"] != primitive:
                raise ValueError(f"Expected {primitive!r} examples, got {example['id']!r}")
            _, questions, media = render(
                processor=processor, state=example["state"],
                questions={"benchmark_test": example["question"]},
                labels=labels, style="chat",
            )
            if media:
                raise ValueError("This training script supports text-only JevBench examples")
            prompt, keys = questions["benchmark_test"]
            target = example["label"]
            if primitive == "noul":
                target = {0: "false", 1: "true"}[target]
            elif primitive == "score":
                target = str(target)
            prompts.append(prompt)
            targets.append(keys.index(target))
            option_counts.append(len(keys))
        encoded = tokenizer(prompts, add_special_tokens=False, return_attention_mask=False)
        return {"input_ids": encoded["input_ids"], "target_indices": targets,
                "num_options": option_counts}

    encoded = dataset.map(
        encode, batched=True, batch_size=256, remove_columns=dataset.column_names,
        # Processor/template changes should never silently reuse old tokens.
        load_from_cache_file=False, desc=f"Render/tokenize {primitive}",
    )
    lengths = [len(ids) for ids in encoded["input_ids"]]
    print(f"Prompt tokens: mean={sum(lengths)/len(lengths):.0f}, max={max(lengths)}")
    return encoded


class JevBenchCollator:
    def __init__(self, tokenizer, option_token_ids):
        self.tokenizer = tokenizer
        self.option_token_ids = torch.tensor(option_token_ids, dtype=torch.long)

    def __call__(self, rows):
        encoded = self.tokenizer.pad(
            {"input_ids": [row["input_ids"] for row in rows]},
            padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        return {**encoded,
                "target_indices": torch.tensor([row["target_indices"] for row in rows]),
                "option_token_ids": self.option_token_ids,
                # Exclude nonexistent options if rows have different counts.
                "option_mask": torch.arange(len(self.option_token_ids))[None, :]
                               < torch.tensor([row["num_options"] for row in rows])[:, None]}


def decision_logits(model, input_ids, attention_mask, option_token_ids):
    # Keep the LoRA modules in the backbone, but avoid allocating logits for
    # the full vocabulary at every prompt position.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.config.model_type != "qwen3_5":
        raise ValueError("Selective decision projection requires the Qwen3.5 JPT architecture")
    hidden = base.model(
        input_ids=input_ids, attention_mask=attention_mask,
        use_cache=False, return_dict=True,
    ).last_hidden_state
    last = attention_mask.sum(dim=1) - 1  # right padding
    hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    head = base.get_output_embeddings()
    weight = head.weight.index_select(0, option_token_ids)
    bias = head.bias.index_select(0, option_token_ids) if head.bias is not None else None
    return F.linear(hidden, weight, bias).float()


def forward_batch(model, batch, primitive="choice"):
    tensors = {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}
    with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
        option_logits = decision_logits(
            model, tensors["input_ids"], tensors["attention_mask"], tensors["option_token_ids"],
        )
    option_logits = option_logits.masked_fill(~tensors["option_mask"], float("-inf"))
    probabilities = F.softmax(option_logits, dim=-1)
    targets = F.one_hot(tensors["target_indices"], num_classes=probabilities.shape[-1]).float()
    per_example_loss = (probabilities - targets).pow(2).sum(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    if primitive == "score":
        # Match llm2jev's expected score and the benchmark's rounding.
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

def evaluate(
    model,
    dataloader,
    primitive="choice",
):
    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_examples = 0

    with torch.inference_mode():

        for batch in dataloader:

            result = forward_batch(
                model,
                batch,
                primitive,
            )

            batch_size = (
                result[
                    "target_indices"
                ]
                .shape[0]
            )

            total_loss += (
                result[
                    "per_example_loss"
                ]
                .sum()
                .item()
            )

            total_correct += (
                result[
                    "correct_mask"
                ]
                .sum()
                .item()
            )

            total_examples += (
                batch_size
            )

    model.train()

    return {
        "loss":
            total_loss
            / total_examples,

        "accuracy":
            total_correct
            / total_examples,

        "correct":
            total_correct,

        "n":
            total_examples,
    }


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check(
    model,
    dataloader,
    primitive="choice",
):
    print()
    print(
        "============================"
    )
    print(
        "BATCH SANITY CHECK"
    )
    print(
        "============================"
    )

    batch = next(
        iter(dataloader)
    )

    print(
        "input_ids:",
        batch[
            "input_ids"
        ].shape,
    )

    print(
        "attention_mask:",
        batch[
            "attention_mask"
        ].shape,
    )

    print(
        "targets:",
        batch[
            "target_indices"
        ].shape,
    )

    print(
        "option tokens:",
        batch[
            "option_token_ids"
        ].shape,
    )

    model.eval()

    with torch.inference_mode():

        result = forward_batch(
            model,
            batch,
            primitive,
        )

    model.train()

    print(
        "probabilities:",
        result[
            "probabilities"
        ].shape,
    )

    print(
        "batch loss:",
        result[
            "loss"
        ].item(),
    )

    probability_sums = (
        result[
            "probabilities"
        ]
        .sum(dim=-1)
    )

    print(
        "probability sums:",
        probability_sums,
    )

    # --------------------------------------------------------
    # REQUIRED CONDITIONS
    # --------------------------------------------------------

    assert (
        result[
            "probabilities"
        ].shape[1]
        ==
        batch["option_token_ids"].numel()
    )

    assert torch.allclose(
        probability_sums,
        torch.ones_like(
            probability_sums
        ),
        atol=1e-4,
    )

    print(
        "Sanity check passed."
    )


# ============================================================
# SAVE
# ============================================================

def save_lora(
    model,
    processor,
    tokenizer,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_dir
    )

    processor.save_pretrained(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print()
    print(
        f"Saved LoRA adapter to "
        f"{output_dir}"
    )


def merge_and_save(
    model,
    processor,
    tokenizer,
    output_dir,
):
    print()
    print(
        "Merging LoRA..."
    )

    merged_model = (
        model.merge_and_unload()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    processor.save_pretrained(
        output_dir
    )

    tokenizer.save_pretrained(
        output_dir
    )

    print(
        f"Saved merged model to "
        f"{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    import importlib.util
    import time

    parser = argparse.ArgumentParser(description="Train JPT on a JevBench dataset with LoRA and Brier loss")
    parser.add_argument("--dataset", choices=DATASETS, default="banking77")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Root for dataset-specific checkpoints")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Physical GPU microbatch")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=importlib.util.find_spec("fla") is None,
        help="Trade speed for VRAM (default: enabled without FLA kernels)",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--validation-samples", type=int, default=VALIDATION_SAMPLES)
    parser.add_argument("--max-steps", type=int, help="Limit microbatches per epoch for speed measurements")
    parser.add_argument("--no-save", action="store_true", help="For speed measurements; skip checkpoint writes")
    args = parser.parse_args()
    if min(args.batch_size, args.grad_accum_steps, args.epochs) < 1:
        parser.error("Batch size, accumulation steps and epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.validation_samples < 1:
        parser.error("--validation-samples must be positive")
    dataset_config = DATASETS[args.dataset]
    primitive = dataset_config["primitive"]
    lora_output_dir = args.output_dir / f"jpt-{args.dataset}-lora"
    merged_output_dir = args.output_dir / f"jpt-{args.dataset}-merged"
    print(f"Dataset={args.dataset}, primitive={primitive}")

    print(f"Device={DEVICE}, dtype={DTYPE}, microbatch={args.batch_size}, "
          f"effective_batch={args.batch_size * args.grad_accum_steps}, "
          f"gradient_checkpointing={args.gradient_checkpointing}")
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name())
    for package in ("fla", "causal_conv1d"):
        print(f"Optional kernel package {package}: {importlib.util.find_spec(package) is not None}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = dataset_config["train"]()
    validation_dataset = dataset_config["validation"]()
    validation_dataset = validation_dataset.select(range(min(args.validation_samples, len(validation_dataset))))
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Training and validation datasets must be nonempty")
    # The prompt renderer defines each primitive's option ordering and count.
    num_options = max(
        len(options_of(json.loads(question))[0])
        for dataset in (train_dataset, validation_dataset)
        for question in dataset["question"]
    )
    labels, token_ids = build_llm2jev_labels(processor, tokenizer, num_options)
    train_dataset = prepare_dataset(train_dataset, processor, tokenizer, labels, primitive)
    validation_dataset = prepare_dataset(validation_dataset, processor, tokenizer, labels, primitive)
    collator = JevBenchCollator(tokenizer, token_ids)
    # Pretokenized rows are cheap to collate; no Windows worker spawn overhead.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, pin_memory=DEVICE.type == "cuda")
    validation_loader = DataLoader(validation_dataset, batch_size=VALIDATION_BATCH_SIZE,
                                   collate_fn=collator, pin_memory=DEVICE.type == "cuda")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, attn_implementation="sdpa",
    ).to(DEVICE)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=find_lora_target_modules(model),
    ))
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sanity_check(model, train_loader, primitive)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=LEARNING_RATE, fused=DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and DTYPE == torch.float16)
    run_started = time.perf_counter()  # Time all training epochs and validation, excluding setup and checkpoint saving.
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = min(len(train_loader), args.max_steps or len(train_loader))
        total_loss = torch.zeros((), device=DEVICE)
        total_correct = torch.zeros((), device=DEVICE, dtype=torch.long)
        seen = tokens = window_examples = updates = 0
        recent_losses, recent_correct = deque(maxlen=ROLLING_WINDOW), deque(maxlen=ROLLING_WINDOW)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for step, batch in enumerate(train_loader):
            if step >= steps:
                break
            result = forward_batch(model, batch, primitive)
            n = batch["input_ids"].shape[0]
            # Sum example losses; normalize gradients once per actual effective
            # batch. This also handles a short final microbatch/window correctly.
            scaler.scale(result["loss"] * n).backward()
            window_examples += n
            boundary = (step + 1) % args.grad_accum_steps == 0 or step + 1 == steps
            if boundary:
                scaler.unscale_(optimizer)
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(window_examples)
                torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                window_examples = 0
                updates += 1
            total_loss += result["per_example_loss"].sum()
            total_correct += result["correct_mask"].sum()
            seen += n
            tokens += int(batch["attention_mask"].sum())
            # Only copy metrics to CPU at logging boundaries.
            recent_losses.append(result["per_example_loss"])
            recent_correct.append(result["correct_mask"])
            if (step + 1) % PRINT_EVERY_BATCHES == 0 or step + 1 == steps:
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                rolling_loss = torch.cat(list(recent_losses))[-ROLLING_WINDOW:].mean().item()
                rolling_acc = torch.cat(list(recent_correct))[-ROLLING_WINDOW:].float().mean().item()
                remaining = (steps - step - 1) * elapsed / (step + 1)
                print(f"epoch={epoch+1} batch={step+1}/{steps} updates={updates} "
                      f"loss={total_loss.item()/seen:.6f} acc={total_correct.item()/seen:.4f} "
                      f"rolling_loss={rolling_loss:.6f} rolling_acc={rolling_acc:.4f} "
                      f"examples/s={seen/elapsed:.2f} tokens/s={tokens/elapsed:.0f} "
                      f"train_ETA_min={remaining/60:.1f}")
        if DEVICE.type == "cuda":
            print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        validation = evaluate(model, validation_loader, primitive)
        print(f"Validation: loss={validation['loss']:.6f}, accuracy={validation['accuracy']:.4f}")
    run_seconds = time.perf_counter() - run_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "lora_run_times.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "dataset": args.dataset, "model": MODEL_ID, "batch_size": args.batch_size,
            "epochs": args.epochs, "grad_accum_steps": args.grad_accum_steps,
            "max_steps_per_epoch": args.max_steps, "gradient_checkpointing": args.gradient_checkpointing,
            "elapsed_seconds": round(run_seconds, 3), "elapsed_minutes": round(run_seconds / 60, 2),
            "scope": "training_and_validation", "approximate": False,
        }) + "\n")
    print(f"LoRA training + validation: {run_seconds / 60:.2f} minutes (logged in {args.output_dir / 'lora_run_times.jsonl'})")
    if not args.no_save:
        save_lora(model, processor, tokenizer, lora_output_dir)
        merge_and_save(model, processor, tokenizer, merged_output_dir)


if __name__ == "__main__":
    main()
