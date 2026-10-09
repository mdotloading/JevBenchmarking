"""What every local decision model (JPT, Decision 2.0) shares: device, dtype and predict()."""
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


class LocalDecisionModel:
    """A model that scores the options of a question in one forward pass. Subclasses implement:

    encode(example, shuffle_seed=None) -> item with "keys" (option keys), "gold_index", "n_tokens"
    collate(items)                     -> batch
    option_logits(batch)               -> (logits [batch, options], option_mask, gold_index), padded options -inf
    add_lora(rank, alpha, dropout)     -> LoRA target module names; makes self.network trainable
    save_adapter(directory)
    and set self.network (the torch module).
    """

    def predict(self, example):
        item = self.encode(example)
        with torch.inference_mode():
            logits, _, _ = self.option_logits(self.collate([item]))
        probabilities = F.softmax(logits[0, :len(item["keys"])], dim=-1)
        return {"keys": item["keys"], "probabilities": probabilities.tolist(), "input_tokens": item["n_tokens"],
                "cost": 0.0}
