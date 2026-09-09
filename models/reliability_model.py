"""Point backbone plus a query-failure reliability head."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from .hug_model import HUGModel
from modules.reliability import ReliabilityHead


class ReliabilityAwareCIR(nn.Module):
    """Reliability-aware CIR without HUG uncertainty estimators in the forward path."""

    def __init__(self, point_model: HUGModel, hidden_dim: int = 768,
                 bottleneck_dim: int = 256, dropout: float = 0.1,
                 freeze_backbone: bool = True):
        super().__init__()
        self.point_model = point_model
        self.reliability_head = ReliabilityHead(hidden_dim, bottleneck_dim, dropout)
        self.freeze_backbone = freeze_backbone
        for name, parameter in self.point_model.named_parameters():
            if freeze_backbone or "uncertainty_estimator" in name or "dynamic_weighting" in name:
                parameter.requires_grad = False

    @classmethod
    def from_point_checkpoint(cls, checkpoint_path: str, device: torch.device,
                              bottleneck_dim: int = 256, dropout: float = 0.1,
                              freeze_backbone: bool = True) -> "ReliabilityAwareCIR":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint.get("args", {})
        recipe = args.get("recipe", "point")
        if recipe not in {"point", "point_continued", "point_robust"}:
            raise ValueError(f"Reliability backbone must be a Point checkpoint, got recipe={recipe!r}")
        point_model = HUGModel(
            num_queries=args.get("num_queries", 32), hidden_dim=args.get("hidden_dim", 768),
            blip_model_name=args.get("blip_model", "pretrain"), freeze_vision_encoder=True,
            text_feature_mode=args.get("text_feature_mode", "query_tokens"),
            uncertainty_is_variance=True, backbone_device=str(device),
        ).to(device)
        point_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = cls(point_model, args.get("hidden_dim", 768), bottleneck_dim, dropout, freeze_backbone)
        model.point_checkpoint = str(Path(checkpoint_path).resolve())
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.point_model.eval()
        return self

    def encode_query_with_reliability(self, ref_pixel_values: torch.Tensor,
                                      text_input_ids: torch.Tensor,
                                      text_attention_mask: torch.Tensor):
        mu_q, _ = self.point_model.encode_query(
            ref_pixel_values, text_input_ids, text_attention_mask, compute_uncertainty=False,
        )
        return mu_q, self.reliability_head(mu_q)

    def forward(self, ref_pixel_values: torch.Tensor, text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                target_pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.point_model(
            ref_pixel_values, text_input_ids, text_attention_mask,
            target_pixel_values, compute_uncertainty=False,
        )
        outputs["failure_logit"] = self.reliability_head(outputs["mu_q"])
        return outputs

    def checkpoint_payload(self, args: Dict, epoch: int, metrics: Dict) -> Dict:
        return {
            "kind": "reliability_aware_cir", "epoch": epoch,
            "model_state_dict": self.state_dict(), "args": args, "metrics": metrics,
            "point_checkpoint": self.point_checkpoint,
        }


def load_reliability_checkpoint(checkpoint_path: str, device: torch.device) -> ReliabilityAwareCIR:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("kind") != "reliability_aware_cir":
        raise ValueError(f"Not a reliability-aware checkpoint: {checkpoint_path}")
    args = checkpoint.get("args", {})
    model = ReliabilityAwareCIR.from_point_checkpoint(
        checkpoint["point_checkpoint"], device,
        bottleneck_dim=args.get("bottleneck_dim", 256), dropout=args.get("dropout", 0.1),
        freeze_backbone=args.get("freeze_backbone", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model
