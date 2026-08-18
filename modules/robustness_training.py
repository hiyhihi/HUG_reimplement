"""Training-time corruptions and calibration losses for RC-HUG (U1–U3)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def uncertainty_scalar(uncertainty: torch.Tensor, uncertainty_is_variance: bool = True) -> torch.Tensor:
    """Return one variance-like uncertainty value per sample."""
    value = uncertainty if uncertainty_is_variance else uncertainty.square()
    return value.float().mean(dim=(-2, -1))


def monotonic_ranking_loss(
    low_uncertainty: torch.Tensor,
    high_uncertainty: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Enforce U(high severity) >= U(low severity) + margin per sample."""
    if low_uncertainty.shape != high_uncertainty.shape:
        raise ValueError("Low/high uncertainty tensors must have identical shape")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return F.relu(float(margin) + low_uncertainty - high_uncertainty).mean()


def _content_token_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask valid text content while preserving first/last special tokens."""
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    lengths = attention_mask.long().sum(dim=1, keepdim=True)
    return attention_mask.bool() & (positions > 0) & (positions < (lengths - 1))


def _apply_text_dropout_from_noise(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probability: float,
    noise: torch.Tensor,
    pad_token_id: int = 0,
    row_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Text-dropout probability must be in [0, 1]")
    if noise.shape != input_ids.shape:
        raise ValueError("noise and input_ids must have the same shape")
    eligible = _content_token_mask(attention_mask)
    if row_mask is not None:
        if row_mask.shape != (input_ids.size(0),):
            raise ValueError("row_mask must have shape [batch]")
        eligible &= row_mask[:, None].bool()
    dropped = eligible & (noise < probability)
    ids = input_ids.clone()
    masks = attention_mask.clone()
    ids[dropped] = pad_token_id
    masks[dropped] = 0
    return ids, masks


def text_dropout_pair(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    low_probability: float,
    high_probability: float,
    pad_token_id: int = 0,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Create nested low/high text corruptions for the same query.

    A shared random matrix makes every token removed at low severity also removed
    at high severity. This provides a meaningful per-query monotonic target.
    """
    if not 0.0 <= low_probability < high_probability <= 1.0:
        raise ValueError("Require 0 <= low_probability < high_probability <= 1")
    noise = torch.rand(input_ids.shape, device=input_ids.device)
    low = _apply_text_dropout_from_noise(input_ids, attention_mask, low_probability, noise, pad_token_id)
    high = _apply_text_dropout_from_noise(input_ids, attention_mask, high_probability, noise, pad_token_id)
    return low, high


def blur_images(images: torch.Tensor, severity: float) -> torch.Tensor:
    """Deterministic, differentiable box-blur used only for U1 calibration views."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("Image severity must be in [0, 1]")
    if severity == 0.0:
        return images
    level = min(4, max(1, math.ceil(severity * 4)))
    kernel = 2 * level + 1
    return F.avg_pool2d(images, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)


def calibration_pair(
    ref_images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    modality: str,
    low_severity: float,
    high_severity: float,
    pad_token_id: int = 0,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return paired low/high corruption views for one modality."""
    if modality == "image":
        return (
            (blur_images(ref_images, low_severity), input_ids, attention_mask),
            (blur_images(ref_images, high_severity), input_ids, attention_mask),
        )
    if modality == "text":
        (low_ids, low_mask), (high_ids, high_mask) = text_dropout_pair(
            input_ids, attention_mask, low_severity, high_severity, pad_token_id
        )
        return (
            (ref_images, low_ids, low_mask),
            (ref_images, high_ids, high_mask),
        )
    raise ValueError("modality must be 'image' or 'text'")


def apply_modality_dropout(
    ref_images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probability: float,
    text_dropout_rate: float,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Drop either reference-image information or text information per selected sample.

    This is U2's training augmentation. It never changes original dataset files.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("modality-dropout probability must be in [0, 1]")
    if not 0.0 <= text_dropout_rate <= 1.0:
        raise ValueError("text-dropout rate must be in [0, 1]")
    if probability == 0.0:
        return ref_images, input_ids, attention_mask, {"image_fraction": 0.0, "text_fraction": 0.0}

    batch_size = ref_images.size(0)
    selected = torch.rand(batch_size, device=ref_images.device) < probability
    image_rows = selected & (torch.rand(batch_size, device=ref_images.device) < 0.5)
    text_rows = selected & ~image_rows

    dropped_images = ref_images.clone()
    dropped_images[image_rows] = 0.0
    noise = torch.rand(input_ids.shape, device=input_ids.device)
    dropped_ids, dropped_masks = _apply_text_dropout_from_noise(
        input_ids, attention_mask, text_dropout_rate, noise, pad_token_id, text_rows
    )
    return dropped_images, dropped_ids, dropped_masks, {
        "image_fraction": image_rows.float().mean().item(),
        "text_fraction": text_rows.float().mean().item(),
    }


def mean_similarity_logits(mu_query: torch.Tensor, mu_target: torch.Tensor, temperature: float) -> torch.Tensor:
    """In-batch mean-retrieval logits used by the clean teacher consistency loss."""
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    q = F.normalize(mu_query.float(), dim=-1)
    c = F.normalize(mu_target.float(), dim=-1)
    return torch.einsum("bkd,ckd->bc", q, c) / (q.size(1) * float(temperature))


def clean_teacher_kl(
    student_query: torch.Tensor,
    student_target: torch.Tensor,
    teacher_query: torch.Tensor,
    teacher_target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher clean retrieval distribution || student corrupted distribution)."""
    student_logits = mean_similarity_logits(student_query, student_target, temperature)
    with torch.no_grad():
        teacher_probs = F.softmax(mean_similarity_logits(teacher_query, teacher_target, temperature), dim=1)
    return F.kl_div(F.log_softmax(student_logits, dim=1), teacher_probs, reduction="batchmean")
