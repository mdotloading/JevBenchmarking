"""Decision-2.0-Eos-0.8B: a Qwen3.5 backbone plus a candidate head that scores every option directly.

The Hub wrapper is inference-only, so we use the torch module behind it (wrapper.decision) with the model's
own vendored prompt encoder and collator (decision2/_vendor/dev2model). Without an adapter this reproduces
system_one exactly. Decision2(None) = the release, Decision2("checkpoints/decision2-x-lora") = LoRA + tuned head.
"""
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModel

from src.data.jevbench import gold_key, shuffle_options
from src.models.base import DEVICE, LocalDecisionModel

MODEL_ID = "vllm-sr/Decision-2.0-Eos-0.8B"
# Last release whose files match MODEL_MANIFEST.json. main (16f56a8, 2026-10-08) edited README.md
# without regenerating the manifest, so the loader's integrity check rejects it.
MODEL_REVISION = "3594047d69f476f1d01cf84c593e213fc3a4dfe0"
HEAD_FILE = "decision_head.safetensors"


class _PosixFingerprintPaths(importlib.abc.MetaPathFinder):
    """Windows workaround for a bug in the Decision 2.0 runtime.

    Its identity check hashes {relative file path: sha256}, building the paths with str(Path), which gives
    'backbone\\model.safetensors' on Windows instead of the 'backbone/model.safetensors' in MODEL_MANIFEST.json,
    so the hash never matches. This rehashes with '/' paths; every file is still verified.
    """

    def find_spec(self, name, path, target=None):
        if not name.endswith("._vendor.dev2model.infer"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_patched(module):
            exec_module(module)
            original = module.checkpoint_fingerprint

            def checkpoint_fingerprint(*args, **kwargs):
                result = original(*args, **kwargs)
                hashes = {key.replace("\\", "/"): value for key, value in result["files_sha256"].items()}
                digest = hashlib.sha256(module.canonical(hashes).encode("utf-8")).hexdigest()
                return {**result, "model_sha256": digest, "files_sha256": hashes}

            module.checkpoint_fingerprint = checkpoint_fingerprint

        spec.loader.exec_module = exec_patched
        return spec


if os.name == "nt" and not any(isinstance(finder, _PosixFingerprintPaths) for finder in sys.meta_path):
    sys.meta_path.insert(0, _PosixFingerprintPaths())


class Decision2(LocalDecisionModel):
    def __init__(self, checkpoint=None):
        # bf16_resident=False keeps FP32 weights and skips the runtime's inference-only fast path.
        wrapper = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, code_revision=MODEL_REVISION,
                                            trust_remote_code=True, bf16_resident=False)
        self.network = wrapper.decision
        self.tokenizer = wrapper.runtime.backend.tokenizer
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.max_tokens = wrapper.max_input_tokens
        vendor = type(self.network).__module__.rsplit(".", 1)[0]
        self.vendored_model = importlib.import_module(f"{vendor}.decision_model")  # encode, collate
        self.vendored_infer = importlib.import_module(f"{vendor}.infer")          # question_to_row
        self.vendored_lora = importlib.import_module(f"{vendor}.lora")            # official LoRA targets
        if checkpoint is not None:
            self.network.backbone = PeftModel.from_pretrained(self.network.backbone, str(checkpoint))
            self.network.head.load_state_dict(load_file(str(Path(checkpoint) / HEAD_FILE)), strict=True)
        self.network.eval()

    def encode(self, example, shuffle_seed=None):
        question = example["question"]
        if shuffle_seed is not None and question["type"] == "choice":
            question = shuffle_options(question, example["id"], shuffle_seed)
        row = self.vendored_infer.question_to_row({"id": example["id"], "state": example["state"]}, "q", question)
        row["label"] = [option["key"] for option in row["options"]].index(gold_key(example))
        item = self.vendored_model.encode(row, self.tokenizer, self.max_tokens)
        return {**item, "gold_index": item["label"], "n_tokens": len(item["ids"])}

    def collate(self, items):
        return self.vendored_model.collate(items, self.pad_id)

    def option_logits(self, batch):
        batch = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            logits = self.network(**batch)  # padded options are already -inf
        return logits.float(), batch["candidate_mask"], batch["labels"]

    def add_lora(self, rank, alpha, dropout):
        """The official layout (vendored lora.py): every layer's MLP + attention/DeltaNet, head fully trained."""
        targets = self.vendored_lora.select_target_modules(self.network.backbone)
        self.network.backbone = get_peft_model(self.network.backbone, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=targets))
        self.network.head.requires_grad_(True)
        return targets

    def enable_gradient_checkpointing(self):
        self.network.backbone.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    def save_adapter(self, directory):
        self.network.backbone.save_pretrained(directory)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.network.head.state_dict().items()},
                  str(Path(directory) / HEAD_FILE))
