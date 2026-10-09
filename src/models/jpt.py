"""JPT (kirp/jpt-0.8b): a chat LM answering with an option label; we read the label-token logits.

JPT(None) = the release, JPT("checkpoints/jpt-x-merged") = a merged fine-tune, JPT("checkpoints/jpt-x-lora") = adapter.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from llm2jev.prompt import find_labels, render
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

from src.data.jevbench import gold_key, shuffle_options
from src.models.base import DEVICE, DTYPE, LocalDecisionModel

MODEL_ID = "kirp/jpt-0.8b"
MAX_OPTIONS = 255
# Linear layers that get LoRA: attention, MLP and Qwen3.5's DeltaNet (vision tower and lm_head stay frozen).
LORA_LAYER_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}


class JPT(LocalDecisionModel):
    def __init__(self, checkpoint=None):
        checkpoint = Path(checkpoint) if checkpoint else None
        is_adapter = checkpoint is not None and (checkpoint / "adapter_config.json").is_file()
        weights = str(checkpoint) if checkpoint is not None and not is_adapter else MODEL_ID
        self.processor = AutoProcessor.from_pretrained(weights)
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.network = AutoModelForMultimodalLM.from_pretrained(
            weights, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)
        self.network.config.use_cache = False
        if is_adapter:
            self.network = PeftModel.from_pretrained(self.network, str(checkpoint))
        self.network.eval()
        # Option labels A, B, ..., AA, ... that are one token after "Answer:"; a question with n options uses the first n.
        _, probe, _ = render(processor=self.processor, state="x", questions={"q": {"type": "noul"}},
                             labels=["A", "B"], style="chat")
        self.labels, self.label_token_ids = find_labels(self.tokenizer, probe["q"][0], n=MAX_OPTIONS)

    def encode(self, example, shuffle_seed=None):
        question = example["question"]
        if shuffle_seed is not None and question["type"] == "choice":
            question = shuffle_options(question, example["id"], shuffle_seed)
        _, rendered, media = render(processor=self.processor, state=example["state"], questions={"q": question},
                                    labels=self.labels, style="chat")
        if media:
            raise ValueError("Only text examples are supported")
        prompt, keys = rendered["q"]
        input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return {"input_ids": input_ids, "keys": keys, "gold_index": keys.index(gold_key(example)),
                "n_tokens": len(input_ids)}

    def collate(self, items):
        padded = self.tokenizer.pad({"input_ids": [item["input_ids"] for item in items]}, padding=True,
                                    pad_to_multiple_of=8, return_tensors="pt")
        n_options = torch.tensor([len(item["keys"]) for item in items])
        width = int(n_options.max())
        return {**padded, "option_token_ids": torch.tensor(self.label_token_ids[:width]),
                "option_mask": torch.arange(width)[None] < n_options[:, None],
                "gold_index": torch.tensor([item["gold_index"] for item in items])}

    def option_logits(self, batch):
        """Logits of the option-label tokens at the last prompt token (only those rows of the LM head)."""
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        network = self.network.get_base_model() if isinstance(self.network, PeftModel) else self.network
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            hidden = network.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   use_cache=False).last_hidden_state
            last_token = batch["attention_mask"].sum(dim=1) - 1
            hidden = hidden[torch.arange(hidden.shape[0], device=DEVICE), last_token]
            head = network.get_output_embeddings()
            logits = F.linear(hidden, head.weight[batch["option_token_ids"]],
                              head.bias[batch["option_token_ids"]] if head.bias is not None else None)
        logits = logits.float().masked_fill(~batch["option_mask"], float("-inf"))
        return logits, batch["option_mask"], batch["gold_index"]

    def add_lora(self, rank, alpha, dropout):
        targets = sorted(name for name, module in self.network.named_modules()
                         if isinstance(module, nn.Linear) and name.split(".")[-1] in LORA_LAYER_NAMES
                         and not any(skip in name.lower() for skip in ("vision", "visual", "lm_head")))
        self.network = get_peft_model(self.network, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                                                bias="none", target_modules=targets))
        return targets

    def enable_gradient_checkpointing(self):
        self.network.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save_adapter(self, directory):
        self.network.save_pretrained(directory)
