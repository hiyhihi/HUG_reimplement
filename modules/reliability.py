"""Reliability head, loss helpers, and calibration metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityHead(nn.Module):
    """Predict query failure from the deterministic fused Point representation.

    The head is query-only, so its probability is available before searching a
    gallery.  Mean and token dispersion retain more failure signal than mean
    pooling alone while remaining independent of HUG variance heads.
    """

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        if fused_tokens.ndim != 3:
            raise ValueError("fused_tokens must have shape [batch, tokens, hidden]")
        pooled = fused_tokens.float().mean(dim=1)
        dispersion = fused_tokens.float().std(dim=1, unbiased=False)
        features = self.input_norm(torch.cat((pooled, dispersion), dim=-1))
        return self.network(features).squeeze(-1)


def failure_prediction_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def binary_auroc(labels: Iterable[int], scores: Iterable[float]) -> Optional[float]:
    labels, scores = list(labels), list(scores)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    ranks = _average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_auprc(labels: Iterable[int], scores: Iterable[float]) -> Optional[float]:
    labels, scores = list(labels), list(scores)
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and have equal length")
    positives = sum(labels)
    if not positives:
        return None
    # Integrate precision at distinct score thresholds. Grouping ties avoids an
    # optimistic result caused by ordering positive labels first inside a tie.
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    found, average_precision, start = 0, 0.0, 0
    while start < len(ranked):
        score, end = ranked[start][0], start + 1
        while end < len(ranked) and ranked[end][0] == score:
            end += 1
        group_positives = sum(label for _, label in ranked[start:end])
        found += group_positives
        average_precision += (found / end) * group_positives / positives
        start = end
    return average_precision


def expected_calibration_error(
    labels: Iterable[int], probabilities: Iterable[float], num_bins: int = 10,
) -> Dict:
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and have equal length")
    bins, ece = [], 0.0
    for index in range(num_bins):
        lower, upper = index / num_bins, (index + 1) / num_bins
        selected = [
            item for item, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == num_bins - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[item] for item in selected) / len(selected)
        failure_rate = sum(labels[item] for item in selected) / len(selected)
        contribution = len(selected) / len(labels) * abs(failure_rate - confidence)
        ece += contribution
        bins.append({
            "lower": lower, "upper": upper, "count": len(selected),
            "mean_failure_probability": confidence, "observed_failure_rate": failure_rate,
        })
    return {"ece": ece, "num_bins": num_bins, "bins": bins}


def risk_coverage(labels: Iterable[int], failure_probabilities: Iterable[float]) -> Dict:
    """Selective retrieval curve, accepting lowest predicted failure first."""
    labels, probabilities = list(labels), list(failure_probabilities)
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and failure_probabilities must be non-empty and equal length")
    order = sorted(range(len(labels)), key=lambda index: probabilities[index])
    cumulative_failures, risks, curve = 0, [], []
    stride = max(1, len(order) // 20)
    for coverage_count, index in enumerate(order, 1):
        cumulative_failures += labels[index]
        risk = cumulative_failures / coverage_count
        risks.append(risk)
        if coverage_count == len(order) or coverage_count % stride == 0:
            curve.append({"coverage": coverage_count / len(order), "risk": risk})
    return {"aurc": sum(risks) / len(risks), "curve": curve}


def reliability_metrics(labels: Iterable[int], probabilities: Iterable[float], num_bins: int = 10) -> Dict:
    labels, probabilities = list(labels), list(probabilities)
    calibration = expected_calibration_error(labels, probabilities, num_bins)
    selective = risk_coverage(labels, probabilities)
    prevalence = sum(labels) / len(labels)
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)
    return {
        "count": len(labels), "failure_prevalence": prevalence,
        "auroc": binary_auroc(labels, probabilities),
        "auprc": binary_auprc(labels, probabilities),
        "ece": calibration["ece"], "brier": brier,
        "aurc": selective["aurc"], "risk_coverage": selective["curve"],
        "calibration_bins": calibration["bins"],
    }
