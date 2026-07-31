"""
Loss Functions for HUG

This module implements the three loss functions used in HUG:
1. L_HC: Holistic Contrastive Loss (Equation 13)
2. L_FC: Fine-Grained Contrastive Loss (Equation 16)
3. L_Cord: Multi-Modal Coordination Loss (Equation 9)

The total loss is: L_HUG = L_HC + lambda_FC * L_FC + lambda_Cord * L_Cord
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional


# def uncertainty_aware_distance(
#     mu_q: torch.Tensor,
#     sigma_q: torch.Tensor,
#     mu_c: torch.Tensor,
#     sigma_c: torch.Tensor,
#     aggregate: bool = True
# ) -> torch.Tensor:
#     """
#     Compute uncertainty-aware distance according to Equation (15).

#     d(z_q, z_c) = ||Â¼_q - Â¼_c||Â²_F + ||Ã_q||Â²_F + ||Ã_c||Â²_F

#     Args:
#         mu_q: Query mean [batch_size, num_queries, hidden_dim]
#         sigma_q: Query uncertainty [batch_size, num_queries, hidden_dim]
#         mu_c: Target mean [batch_size, num_queries, hidden_dim]
#         sigma_c: Target uncertainty [batch_size, num_queries, hidden_dim]
#         aggregate: If True, sum over all dimensions to get scalar distance

#     Returns:
#         distance: Scalar distance if aggregate=True, otherwise maintains dimensions
#     """
#     # Frobenius norm squared: sum of squared elements
#     # mu_q = F.normalize(mu_q, dim=-1)
#     # mu_c = F.normalize(mu_c, dim=-1)

#     # mean_distance = torch.sum((mu_q - mu_c) ** 2, dim=(-2, -1)) if aggregate else (mu_q - mu_c) ** 2
#     # sigma_q_norm = torch.sum(sigma_q ** 2, dim=(-2, -1)) if aggregate else sigma_q ** 2
#     # sigma_c_norm = torch.sum(sigma_c ** 2, dim=(-2, -1)) if aggregate else sigma_c ** 2
#     # scale = mu_q.size(-1) * mu_q.size(-2)

#     # mean_distance = torch.sum(
#     #     (mu_q - mu_c) ** 2,
#     #     dim=(-2, -1)
#     # ) / scale

#     # sigma_q_norm = torch.sum(
#     #     sigma_q ** 2,
#     #     dim=(-2, -1)
#     # ) / scale

#     # sigma_c_norm = torch.sum(
#     #     sigma_c ** 2,
#     #     dim=(-2, -1)
#     # ) / scale

#     # # distance = mean_distance + sigma_q_norm + sigma_c_norm
#     # distance = mean_distance + sigma_q_norm + sigma_c_norm

#     # return distance
#     mu_q_exp = mu_q.unsqueeze(1)      # [B,1,K,D]
#     mu_c_exp = mu_c.unsqueeze(0)      # [1,B,K,D]

#     sigma_q_exp = sigma_q.unsqueeze(1)
#     sigma_c_exp = sigma_c.unsqueeze(0)

#     mean_distance = ((mu_q_exp - mu_c_exp) ** 2).sum(dim=(-2,-1))
#     sigma_q_norm = (sigma_q_exp ** 2).sum(dim=(-2,-1))
#     sigma_c_norm = (sigma_c_exp ** 2).sum(dim=(-2,-1))

#     distances = mean_distance + sigma_q_norm + sigma_c_norm

#     return distances

# class HolisticContrastiveLoss(nn.Module):
#     """
#     Holistic Contrastive Loss (L_HC) using Sigmoid loss.

#     According to Equation (13):
#     L_HC = -log(Ã(-aÂ·d_pos - b)) - Â£ log(Ã(aÂ·d_neg + b))

#     where a and b are learnable parameters, and Ã is the sigmoid function.
#     """

#     def __init__(self, init_a: float = 0.5, init_b: float = 0.0):
#         """
#         Args:
#             init_a: Initial value for learnable parameter a
#             init_b: Initial value for learnable parameter b
#         """
#         super().__init__()
#         self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
#         self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
#         # self.register_buffer('a', torch.tensor(init_a))
#         # self.register_buffer('b', torch.tensor(init_b))

#     def forward(
#         self,
#         mu_q: torch.Tensor,
#         sigma_q: torch.Tensor,
#         mu_c: torch.Tensor,
#         sigma_c: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Compute holistic contrastive loss.

#         Args:
#             mu_q: Query mean [batch_size, num_queries, hidden_dim]
#             sigma_q: Query uncertainty [batch_size, num_queries, hidden_dim]
#             mu_c: Target mean [batch_size, num_queries, hidden_dim]
#             sigma_c: Target uncertainty [batch_size, num_queries, hidden_dim]

#         Returns:
#             loss: Scalar loss value
#         """
#         batch_size = mu_q.size(0)
#         # Compute pairwise distances between all queries and targets in the batch
#         # Shape: [batch_size, batch_size]
#         distances = torch.zeros(batch_size, batch_size, device=mu_q.device)

#         for i in range(batch_size):
#             for j in range(batch_size):
#                 distances[i, j] = uncertainty_aware_distance(
#                     mu_q[i:i+1], sigma_q[i:i+1],
#                     mu_c[j:j+1], sigma_c[j:j+1],
#                     aggregate=True
#                 )

#         # Extract positive distances (diagonal elements)
#         positive_distances = torch.diag(distances)
        
#         # Loss for positive pairs
#         loss_pos = -torch.log(torch.sigmoid(-self.a * positive_distances - self.b) + 1e-8)

#         # Loss for negative pairs (all off-diagonal elements)
#         loss_neg = 0
#         for i in range(batch_size):
#             # Get all negative distances for query i
#             neg_mask = torch.ones(batch_size, dtype=torch.bool, device=mu_q.device)
#             neg_mask[i] = False
#             negative_distances = distances[i][neg_mask]

#             # Sum over all negatives
#             loss_neg += -torch.sum(torch.log(torch.sigmoid(self.a * negative_distances + self.b) + 1e-8))

#         # Average over batch
#         loss = (torch.sum(loss_pos) + loss_neg) / batch_size

        
#         # cos_tokens = F.cosine_similarity(
#         #     mu_q[:, 0, :],
#         #     mu_q[:, 1, :],
#         #     dim=-1
#         # )

#         # print("token cosine:", cos_tokens.mean().item())

#         # sample_cos = F.cosine_similarity(
#         #     mu_q[0].flatten(),
#         #     mu_q[1].flatten(),
#         #     dim=0
#         # )

#         # print("sample cosine:", sample_cos.item())

#         print("mu_q token std:", mu_q.std(dim=1).mean().item())
#         print("mu_c token std:", mu_c.std(dim=1).mean().item())
#         inbatch_sim = torch.matmul(
#             F.normalize(mu_q.mean(1), dim=-1),
#             F.normalize(mu_q.mean(1), dim=-1).T
#         )

#         print(inbatch_sim.mean())
#         print(inbatch_sim.std())
#         # DEBUG
#         with torch.no_grad():
#             print("\n===== HC DEBUG =====")
#             print(f"pos_dist mean: {positive_distances.mean().item():.4f}")
#             print(f"pos_dist min : {positive_distances.min().item():.4f}")
#             print(f"pos_dist max : {positive_distances.max().item():.4f}")

#             # lấy toàn bộ negative distances
#             neg_values = distances[~torch.eye(batch_size, dtype=torch.bool, device=distances.device)]

#             if neg_values.numel() > 0:

#                 print(f"neg_dist mean: {neg_values.mean().item():.4f}")
#                 print(f"neg_dist min : {neg_values.min().item():.4f}")
#                 print(f"neg_dist max : {neg_values.max().item():.4f}")

#             print(f"a: {self.a.item():.4f}")
#             print(f"b: {self.b.item():.4f}")

#             print(f"loss_hc: {loss.item():.4f}")

#         return loss

def _as_variance(uncertainty: torch.Tensor, uncertainty_is_variance: bool) -> torch.Tensor:
    """Convert a legacy standard deviation or a paper-style variance to variance."""
    return uncertainty if uncertainty_is_variance else uncertainty.square()


def pairwise_uncertainty_distance(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_c: torch.Tensor,
    sigma_c: torch.Tensor,
    uncertainty_is_variance: bool = False,
    include_uncertainty: bool = True,
) -> torch.Tensor:
    """Vectorized normalized version of Eq. 15 for arbitrary Q x C inputs."""
    if mu_q.ndim != 3 or mu_c.ndim != 3:
        raise ValueError("mu_q and mu_c must have shape [N, K, D]")
    if mu_q.shape[1:] != mu_c.shape[1:]:
        raise ValueError("query and candidate feature shapes must match")
    q = F.normalize(mu_q.float(), p=2, dim=-1)
    c = F.normalize(mu_c.float(), p=2, dim=-1)
    num_components = q.size(1)
    similarity = q.flatten(1) @ c.flatten(1).T / num_components
    mean_distance = (2.0 - 2.0 * similarity).clamp_min(0.0)
    if not include_uncertainty:
        return mean_distance
    var_q = _as_variance(sigma_q.float(), uncertainty_is_variance).mean(dim=(-2, -1))
    var_c = _as_variance(sigma_c.float(), uncertainty_is_variance).mean(dim=(-2, -1))
    return mean_distance + var_q[:, None] + var_c[None, :]


def uncertainty_aware_distance(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    mu_c: torch.Tensor,
    sigma_c: torch.Tensor,
    aggregate: bool = True,
    uncertainty_is_variance: bool = False,
    include_uncertainty: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Backward-compatible public distance function."""
    del aggregate, kwargs
    return pairwise_uncertainty_distance(
        mu_q, sigma_q, mu_c, sigma_c,
        uncertainty_is_variance=uncertainty_is_variance,
        include_uncertainty=include_uncertainty,
    )


class HolisticContrastiveLoss(nn.Module):
    """Sigmoid holistic contrast with legacy and paper-compatible variants."""
    def __init__(
        self, init_a=2.0, init_b=-5.0, symmetric=False,
        uncertainty_is_variance=False,
    ):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.symmetric = symmetric
        self.uncertainty_is_variance = uncertainty_is_variance

    def forward(self, mu_q, sigma_q, mu_c, sigma_c):
        distances = pairwise_uncertainty_distance(
            mu_q, sigma_q, mu_c, sigma_c,
            uncertainty_is_variance=self.uncertainty_is_variance,
        )
        batch_size = distances.size(0)
        if batch_size < 2:
            raise ValueError("Holistic contrast requires batch_size >= 2")
        positive_loss = F.softplus(self.a * distances.diagonal() + self.b).mean()
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=distances.device)
        negative_loss = F.softplus(-self.a * distances - self.b).masked_fill(~mask, 0)
        row_loss = negative_loss.sum(dim=1).mean()
        if not self.symmetric:
            return positive_loss + row_loss
        return positive_loss + row_loss + negative_loss.sum(dim=0).mean()


class PointContrastiveLoss(nn.Module):
    """Symmetric InfoNCE baseline on the 32 aligned Q-Former mean tokens."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, mu_q, mu_c):
        q = F.normalize(mu_q.float(), dim=-1)
        c = F.normalize(mu_c.float(), dim=-1)
        logits = torch.einsum('bkd,ckd->bc', q, c) / (q.size(1) * self.temperature)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class LegacyFineGrainedContrastiveLoss(nn.Module):
    """Vectorized equivalent of the historical InfoNCE uncertainty loss."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sigma_q, sigma_c):
        q = F.normalize(sigma_q.float(), dim=-1)
        c = F.normalize(sigma_c.float(), dim=-1)
        batch_size, num_components, _ = q.shape
        pos = (q * c).sum(dim=-1, keepdim=True)
        same_q = torch.einsum('bkd,bjd->bkj', q, q)
        same_c = torch.einsum('bkd,bjd->bkj', q, c)
        offdiag = ~torch.eye(num_components, dtype=torch.bool, device=q.device)
        neg_q = same_q[:, offdiag].view(batch_size, num_components, num_components - 1)
        neg_c = same_c[:, offdiag].view(batch_size, num_components, num_components - 1)
        logits = torch.cat([pos, neg_q, neg_c], dim=-1) / self.temperature
        component_loss = F.cross_entropy(
            logits.flatten(0, 1),
            torch.zeros(batch_size * num_components, dtype=torch.long, device=q.device),
        )
        instance_logits = torch.einsum('bkd,ckd->kbc', q, c) / self.temperature
        labels = torch.arange(batch_size, device=q.device).expand(num_components, -1)
        instance_loss = F.cross_entropy(instance_logits.flatten(0, 1), labels.flatten())
        return (batch_size * component_loss + instance_loss) / (batch_size + 1)


FineGrainedContrastiveLoss = LegacyFineGrainedContrastiveLoss


class PaperFineGrainedContrastiveLoss(nn.Module):
    """Eq. 16 using component-, instance-, and modality-wise negatives."""
    def __init__(self, init_a: float = 1.0, init_b: float = 0.0):
        super().__init__()
        self.a_prime = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b_prime = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, sigma_q, sigma_c):
        vectors = torch.cat([sigma_q, sigma_c], dim=0).float().flatten(0, 1)
        vectors = F.normalize(vectors, dim=-1)
        distances = (2.0 - 2.0 * (vectors @ vectors.T)).clamp_min(0.0)
        mask = ~torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        return F.softplus(-self.a_prime * distances[mask] - self.b_prime).mean()


class MultiModalCoordinationLoss(nn.Module):
    """Matched image-text pairs must have lower variance than deranged pairs."""
    def __init__(self, margin=0.2, uncertainty_is_variance=False, use_sigmoid=False):
        super().__init__()
        self.margin = margin
        self.uncertainty_is_variance = uncertainty_is_variance
        self.use_sigmoid = use_sigmoid

    def forward(self, sigma_m_matched, sigma_m_mismatched):
        matched = _as_variance(sigma_m_matched.float(), self.uncertainty_is_variance).mean(dim=(-2, -1))
        mismatched = _as_variance(sigma_m_mismatched.float(), self.uncertainty_is_variance).mean(dim=(-2, -1))
        if self.use_sigmoid:
            return F.softplus(matched - mismatched).mean()
        return F.relu(self.margin + matched - mismatched).mean()


class HUGLoss(nn.Module):
    """Selectable objective for faithful ablations and legacy reproduction."""
    def __init__(
        self, lambda_fc=0.5, lambda_cord=0.1, margin_cord=0.2,
        temperature_fc=0.07, recipe="paper", uncertainty_is_variance=True,
    ):
        super().__init__()
        if recipe not in {"legacy", "point", "paper"}:
            raise ValueError(f"Unknown loss recipe: {recipe}")
        self.recipe = recipe
        self.lambda_fc = lambda_fc
        self.lambda_cord = lambda_cord
        self.needs_mismatched = recipe != "point"
        if recipe == "point":
            self.loss_point = PointContrastiveLoss(temperature_fc)
        elif recipe == "legacy":
            self.loss_hc = HolisticContrastiveLoss(2.0, -5.0, False, False)
            self.loss_fc = LegacyFineGrainedContrastiveLoss(temperature_fc)
            self.loss_cord = MultiModalCoordinationLoss(margin_cord, False, False)
        else:
            self.loss_hc = HolisticContrastiveLoss(1.0, 0.0, True, uncertainty_is_variance)
            self.loss_fc = PaperFineGrainedContrastiveLoss()
            self.loss_cord = MultiModalCoordinationLoss(0.0, uncertainty_is_variance, True)

    def forward(
        self, mu_q, sigma_q, mu_c, sigma_c,
        sigma_m_matched=None, sigma_m_mismatched=None,
    ):
        if self.recipe == "point":
            total = self.loss_point(mu_q, mu_c)
            zero = total.detach().new_zeros(())
            return {'total_loss': total, 'loss_hc': total.detach(), 'loss_fc': zero, 'loss_cord': zero}
        if sigma_m_matched is None or sigma_m_mismatched is None:
            raise ValueError(f"{self.recipe} recipe requires matched and mismatched sigma_m")
        l_hc = self.loss_hc(mu_q, sigma_q, mu_c, sigma_c)
        l_fc = self.loss_fc(sigma_q, sigma_c)
        l_cord = self.loss_cord(sigma_m_matched, sigma_m_mismatched)
        total = l_hc + self.lambda_fc * l_fc + self.lambda_cord * l_cord
        return {
            'total_loss': total, 'loss_hc': l_hc.detach(),
            'loss_fc': l_fc.detach(), 'loss_cord': l_cord.detach(),
        }
