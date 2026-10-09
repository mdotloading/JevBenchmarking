"""Scoring shared by training and the benchmark. Inputs are option probabilities in option-key order."""
import torch
import torch.nn.functional as F

ECE_BINS = 10


def predicted_index(probabilities, modality):
    """The answer: the most likely option, or for score questions the rounded expected level."""
    if modality == "score":
        levels = torch.arange(probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype)
        expected = (probabilities * levels).sum(dim=-1)
        return expected.round().long().clamp(0, probabilities.shape[-1] - 1)
    return probabilities.argmax(dim=-1)


def brier(probabilities, gold_index):
    """Multi-class Brier score per example (0 = certain and right, 2 = certain and wrong)."""
    one_hot = F.one_hot(gold_index, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return (probabilities - one_hot).pow(2).sum(dim=-1)


def expected_calibration_error(confidence, correct, n_bins=ECE_BINS):
    """Top-label ECE: |confidence - accuracy| over equal-width confidence bins, weighted by bin size."""
    confidence, correct = torch.as_tensor(confidence, dtype=torch.float), torch.as_tensor(correct, dtype=torch.float)
    bins = (confidence * n_bins).long().clamp(max=n_bins - 1)
    return sum(((bins == b).float().mean() * (confidence[bins == b].mean() - correct[bins == b].mean()).abs()).item()
               for b in range(n_bins) if (bins == b).any())


def training_objective(logits, gold_index, option_mask, modality, score_sigma=0.0):
    """Per-example loss: Brier, or for score questions with score_sigma > 0 cross-entropy against
    Gaussian soft labels around the gold level (a near miss costs less than a far one)."""
    if modality == "score" and score_sigma:
        levels = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float)
        soft = torch.exp(-(levels[None] - gold_index[:, None].float()).pow(2) / (2 * score_sigma ** 2)) * option_mask
        soft = soft / soft.sum(dim=-1, keepdim=True)
        return -(soft * F.log_softmax(logits, dim=-1).masked_fill(~option_mask, 0.0)).sum(dim=-1)
    return brier(F.softmax(logits, dim=-1), gold_index)
