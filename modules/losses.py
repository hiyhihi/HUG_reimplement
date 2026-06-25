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

def uncertainty_aware_distance(mu_q, sigma_q, mu_c, sigma_c, aggregate=True, **kwargs):
    # 1. BẮT BUỘC: Chuẩn hóa L2 trên chiều D (768) để giới hạn ||mu|| = 1
    mu_q_norm = F.normalize(mu_q, p=2, dim=-1)
    mu_c_norm = F.normalize(mu_c, p=2, dim=-1)

    mu_q_exp = mu_q_norm.unsqueeze(1)    # [B, 1, K, D]
    mu_c_exp = mu_c_norm.unsqueeze(0)    # [1, B, K, D]
    sigma_q_exp = sigma_q.unsqueeze(1)   # [B, 1, K, D]
    sigma_c_exp = sigma_c.unsqueeze(0)   # [1, B, K, D]

    # 2. TÍNH TOÁN KHOẢNG CÁCH CHUẨN MỰC
    # Phía mu: Vẫn dùng .sum(dim=-1) để lấy khoảng cách hình học chuẩn [0, 4]
    mean_distance = ((mu_q_exp - mu_c_exp) ** 2).sum(dim=-1).mean(dim=-1)

    # ĐÂY LÀ ĐIỂM SỬA QUYẾT ĐỊNH: 
    # Phía sigma: BẮT BUỘC dùng .mean(dim=-1) thay vì .sum(dim=-1) 
    # Để tránh việc sigma bị nhân lên 768 lần và đè bẹp mu
    sigma_q_norm_dist = (sigma_q_exp ** 2).mean(dim=-1).mean(dim=-1)
    sigma_c_norm_dist = (sigma_c_exp ** 2).mean(dim=-1).mean(dim=-1)

    # Lưu lại để debug
    global DEBUG_MEAN_DIST, DEBUG_SIGMA_DIST
    DEBUG_MEAN_DIST = mean_distance.detach()
    DEBUG_SIGMA_DIST = (sigma_q_norm_dist + sigma_c_norm_dist).detach()

    return mean_distance + sigma_q_norm_dist + sigma_c_norm_dist

class HolisticContrastiveLoss(nn.Module):
    # Giữ nguyên a=2.0, b=-5.0 vì scale [0, 4] rất khớp với mức này
    def __init__(self, init_a: float = 2.0, init_b: float = -5.0): 
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, mu_q, sigma_q, mu_c, sigma_c):
        batch_size = mu_q.size(0)
        distances = uncertainty_aware_distance(mu_q, sigma_q, mu_c, sigma_c)

        positive_distances = torch.diag(distances)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=distances.device)
        negative_distances = distances[mask].view(batch_size, batch_size - 1)
        
        loss_pos = F.softplus(self.a * positive_distances + self.b) 
        loss_neg_matrix = F.softplus(-self.a * negative_distances - self.b)
        
        # Bắt buộc Sum để cộng hưởng lực đẩy
        loss_neg = loss_neg_matrix.sum(dim=1) 

        loss = (loss_pos + loss_neg).mean()

        with torch.no_grad():
            global DEBUG_MEAN_DIST, DEBUG_SIGMA_DIST
            pos_mean_dist = torch.diag(DEBUG_MEAN_DIST).mean().item()
            pos_sigma_dist = torch.diag(DEBUG_SIGMA_DIST).mean().item()
            
            print(f"\n===== HC DEBUG =====")
            print(f"pos_dist (Total) : {positive_distances.mean().item():.4f}")
            print(f"neg_dist (Total) : {negative_distances.mean().item():.4f}")
            print(f"--> Semantic (mu)  : {pos_mean_dist:.4f} (Max 4.0)")
            print(f"--> Cheat (sigma)  : {pos_sigma_dist:.4f}")
            print(f"a: {self.a.item():.4f} | b: {self.b.item():.4f}")

        return loss

class FineGrainedContrastiveLoss(nn.Module):
    """
    Fine-Grained Contrastive Loss (L_FC).

    According to Equation (16), this loss applies contrastive learning
    to each of the K Gaussian distributions independently, using three
    types of negative sampling:
    - Component-wise: Different k' within the same sample
    - Instance-wise: Same k from different samples
    - Modality-wise: Cross-modal negatives (query vs target)
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        sigma_q: torch.Tensor,
        sigma_c: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fine-grained contrastive loss.

        Args:
            sigma_q: Query uncertainty [batch_size, num_queries, hidden_dim]
            sigma_c: Target uncertainty [batch_size, num_queries, hidden_dim]

        Returns:
            loss: Scalar loss value
        """
        batch_size, num_queries, hidden_dim = sigma_q.shape

        total_loss = 0
        num_positives = 0

        # Iterate over each Gaussian component (k)
        for k in range(num_queries):
            # Get the k-th component for all samples
            sigma_q_k = sigma_q[:, k, :]  # [batch_size, hidden_dim]
            sigma_c_k = sigma_c[:, k, :]  # [batch_size, hidden_dim]

            # Normalize features
            sigma_q_k_norm = F.normalize(sigma_q_k, dim=-1)
            sigma_c_k_norm = F.normalize(sigma_c_k, dim=-1)

            # 1. Component-wise negatives: other k' from the same sample
            # For each sample, contrast k-th component with all other components
            for i in range(batch_size):
                # Anchor: k-th component of query i
                anchor = sigma_q_k_norm[i:i+1]  # [1, hidden_dim]

                # Positive: k-th component of target i
                positive = sigma_c_k_norm[i:i+1]  # [1, hidden_dim]

                # Negatives: all other components from query i and target i
                neg_q = sigma_q[i, :, :]  # [num_queries, hidden_dim]
                neg_c = sigma_c[i, :, :]  # [num_queries, hidden_dim]
                negatives = torch.cat([neg_q, neg_c], dim=0)  # [2*num_queries, hidden_dim]
                negatives_norm = F.normalize(negatives, dim=-1)

                # Remove the positive component from negatives
                # Keep all except k-th from query and k-th from target
                neg_mask = torch.ones(2 * num_queries, dtype=torch.bool, device=sigma_q.device)
                neg_mask[k] = False  # Remove k-th from query side
                neg_mask[num_queries + k] = False  # Remove k-th from target side
                negatives_filtered = negatives_norm[neg_mask]

                # Compute similarities
                pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
                neg_sim = torch.matmul(anchor, negatives_filtered.T) / self.temperature

                # InfoNCE loss
                logits = torch.cat([pos_sim, neg_sim.squeeze(0)])
                labels = torch.zeros(1, dtype=torch.long, device=sigma_q.device)
                loss = F.cross_entropy(logits.unsqueeze(0), labels)

                total_loss += loss
                num_positives += 1

            # 2. Instance-wise negatives: same k from different samples (in-batch)
            # Use standard InfoNCE with batch samples
            similarities = torch.matmul(sigma_q_k_norm, sigma_c_k_norm.T) / self.temperature
            labels = torch.arange(batch_size, device=sigma_q.device)
            loss_instance = F.cross_entropy(similarities, labels)
            total_loss += loss_instance

        # Average over all positives
        avg_loss = total_loss / (num_positives + num_queries)

        return avg_loss


class MultiModalCoordinationLoss(nn.Module):
    """
    Multi-Modal Coordination Loss (L_Cord).

    According to Equation (9), this is a ranking loss that ensures
    matched (image, text) pairs have lower coordination uncertainty
    than mismatched pairs.

    L_Cord = max(0, margin + ||Ã_m(matched)||Â² - ||Ã_m(mismatched)||Â²)
    """

    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margin for ranking loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        sigma_m_matched: torch.Tensor,
        sigma_m_mismatched: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-modal coordination loss.

        Args:
            sigma_m_matched: Coordination uncertainty for matched pairs
                            [batch_size, num_queries, hidden_dim]
            sigma_m_mismatched: Coordination uncertainty for mismatched pairs
                               [batch_size, num_queries, hidden_dim]

        Returns:
            loss: Scalar loss value
        """
        # Compute L2 norm squared of coordination uncertainties
        matched_norm = torch.mean(sigma_m_matched ** 2, dim=(-2, -1))  # [batch_size]
        mismatched_norm = torch.mean(sigma_m_mismatched ** 2, dim=(-2, -1))  # [batch_size]

        # Ranking loss: matched should have lower uncertainty than mismatched
        loss = torch.mean(torch.clamp(self.margin + matched_norm - mismatched_norm, min=0))

        return loss


class HUGLoss(nn.Module):
    """
    Combined HUG loss: L_HUG = L_HC + lambda_FC * L_FC + lambda_Cord * L_Cord

    Args:
        lambda_fc: Weight for fine-grained contrastive loss (default: 0.5)
        lambda_cord: Weight for coordination loss (default: 0.1)
    """

    def __init__(
        self,
        lambda_fc: float = 0.5,
        lambda_cord: float = 0.1,
        margin_cord: float = 0.2,
        temperature_fc: float = 0.07
    ):
        super().__init__()

        self.lambda_fc = lambda_fc
        self.lambda_cord = lambda_cord

        self.loss_hc = HolisticContrastiveLoss()
        self.loss_fc = FineGrainedContrastiveLoss(temperature=temperature_fc)
        self.loss_cord = MultiModalCoordinationLoss(margin=margin_cord)

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_c: torch.Tensor,
        sigma_c: torch.Tensor,
        sigma_m_matched: torch.Tensor,
        sigma_m_mismatched: torch.Tensor
    ) -> dict:
        """
        Compute total HUG loss and individual loss components.

        Args:
            mu_q: Query mean
            sigma_q: Query uncertainty
            mu_c: Target mean
            sigma_c: Target uncertainty
            sigma_m_matched: Coordination uncertainty for matched pairs
            sigma_m_mismatched: Coordination uncertainty for mismatched pairs

        Returns:
            Dictionary containing total loss and individual components
        """
        # Compute individual losses
        l_hc = self.loss_hc(mu_q, sigma_q, mu_c, sigma_c)
        l_fc = self.loss_fc(sigma_q, sigma_c)
        l_cord = self.loss_cord(sigma_m_matched, sigma_m_mismatched)

        # Total loss
        total_loss = l_hc + self.lambda_fc * l_fc + self.lambda_cord * l_cord

        return {
            'total_loss': total_loss,
            'loss_hc': l_hc.detach(),
            'loss_fc': l_fc.detach(),
            'loss_cord': l_cord.detach()
        }
