# Copyright © MERL 2026
# Description: Defines the follower UAV actor network with sensor and upstream attention.
from pathlib import Path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.distributions import VonMises3D
import numpy as np

from vmf3_sampler import sample_vmf3, entropy_vmf3

def _expand_scalar_feature(value, batch_size, device, dtype):
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.view(1, 1)
    elif value.ndim == 1:
        value = value.view(-1, 1)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, -1)
    return value

def _safe_normalize(vec, eps=1e-8):
    return F.normalize(vec, p=2, dim=-1, eps=eps)

def _gather_selected_sensor_pos(sensor_pos, sensor_action):
    batch_idx = torch.arange(sensor_pos.shape[0], device=sensor_pos.device)
    return sensor_pos[batch_idx, sensor_action]


def _has_visible_sensor_rows(sensor_mask, visited):
    if sensor_mask is not None:
        return sensor_mask.bool().any(dim=-1)
    return (~visited.bool()).any(dim=-1)


class SensorFeatureBuilder(nn.Module):
    def __init__(self):
        super(SensorFeatureBuilder, self).__init__()

    def forward(
        self,
        sensor_pos,
        agent_pos,
        u2s_range,
    ):
        """
        nearby_sensor_pos: (B, N, d)
        agent_pos: (B, d)
        visited: (B, N)
        Returns:
        sensor_features: (B, N, 2*d+6)
        """
        # combined_sensor_pos = sensor_pos  # (B, N+M, d)
        agent_pos_expanded = agent_pos.unsqueeze(1) # (B, 1, d)
        rel_sensor_pos = sensor_pos - agent_pos_expanded # (B, N+M, d) relative positions of sensors with respect to the agent
        rel_sensor_dir = _safe_normalize(rel_sensor_pos) # (B, N+M, d) unit direction vectors from agent to sensors
        
        # Ensure u2s_range broadcasts correctly against (B, N, 1).
        u2s = torch.as_tensor(u2s_range, device=sensor_pos.device, dtype=sensor_pos.dtype)
        if u2s.ndim == 0:
            pass  # scalar: broadcasts fine
        elif u2s.ndim == 1:
            u2s = u2s.view(-1, 1, 1)  # (B,) -> (B, 1, 1)
        elif u2s.ndim == 2:
            u2s = u2s.unsqueeze(1)    # (B, 1) -> (B, 1, 1)

        sensor_dist = torch.norm(rel_sensor_pos, dim=-1, keepdim=True) - u2s # (B, N, 1)

        sensor_features = torch.cat([
            agent_pos_expanded.expand(-1, sensor_pos.shape[1], -1), # (B, N, d) absolute agent position
            rel_sensor_dir, # (B, N+M, d) unit direction vectors from agent to sensors    
            sensor_dist, # (B, N+M, 1) Euclidean distance of each sensor from the agent
        ], dim=-1)
        return sensor_features


class SensorEncoder(nn.Module):
    def __init__(self, sensor_feature_dim, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(sensor_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sensor_features):
        sensor_embeddings = self.mlp(sensor_features)
        return sensor_embeddings


# class SensorAttentionEncoder(nn.Module):
#     def __init__(self, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.0):
#         super().__init__()

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=4 * hidden_dim,
#             dropout=dropout,
#             batch_first=True,
#             activation="relu",
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers,
#         )

#     def forward(self, sensor_embeddings, sensor_mask=None):
#         if sensor_mask is not None:
#             padding_mask = ~sensor_mask.bool()
#         else:
#             padding_mask = None

#         return self.transformer(
#             sensor_embeddings,
#             src_key_padding_mask=padding_mask,
#         )


class UpstreamFeatureBuilder(nn.Module):
    def __init__(self):
        super(UpstreamFeatureBuilder, self).__init__()

    def forward(
        self,
        upstream_neighbor_pos,
        upstream_neighbor_vel,
        agent_pos,
        agent_vel,
        self_sensor_coverage,
        remaining_sensor_coverage,
        remaining_time,
        u2u_range,
    ):
        upstream_rel_pos = upstream_neighbor_pos - agent_pos
        upstream_dist = torch.norm(upstream_rel_pos, dim=-1, keepdim=True)
        upstream_dir = _safe_normalize(upstream_rel_pos)

        batch_size = agent_pos.shape[0]
        u2u_range = _expand_scalar_feature(
            u2u_range,
            batch_size=batch_size,
            device=agent_pos.device,
            dtype=agent_pos.dtype,
        )

        upstream_link_margin = u2u_range - upstream_dist
        upstream_rel_vel = upstream_neighbor_vel - agent_vel
        upstream_rel_normalized_vel = _safe_normalize(upstream_rel_vel)

        features = [
            agent_pos,
            upstream_dir,
            upstream_dist,
            upstream_rel_normalized_vel,
            upstream_link_margin,
            self_sensor_coverage.float().view(-1, 1),
            remaining_sensor_coverage.float().view(-1, 1),
            remaining_time.float().view(-1, 1),
        ]

        return torch.cat(features, dim=-1)


class UpstreamEncoder(nn.Module):
    def __init__(self, upstream_feature_dim, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(upstream_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, upstream_features):
        return self.mlp(upstream_features)

class AttentionPooling(nn.Module):
    """
    Performs attention-based pooling over sensor embeddings.
    Computes attention scores for each sensor and produces a single pooled vector.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, sensor_embeddings, sensor_mask=None):
        """
        Args:
            sensor_embeddings: (B, N, hidden_dim) sensor embeddings from attention encoder
            sensor_mask: (B, N) optional mask for sensors that are actually observable
        
        Returns:
            pooled_features: (B, hidden_dim) single pooled vector
        """
        Q = self.query(sensor_embeddings)  # (B, N, hidden_dim)
        K = self.key(sensor_embeddings)    # (B, N, hidden_dim)
        V = self.value(sensor_embeddings)  # (B, N, hidden_dim)
        
        # Self-attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, N, N)
        
        if sensor_mask is not None:
            valid_sensor_mask = sensor_mask.bool()
            scores = scores.masked_fill(~valid_sensor_mask.unsqueeze(1), -1e9)
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, N, N)
        
        if sensor_mask is not None:
            # Exclude padded sensors as both keys and queries.
            key_mask = valid_sensor_mask.unsqueeze(1).to(attn_weights.dtype)
            attn_weights = attn_weights * key_mask
            attn_weights = attn_weights / attn_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)

            query_mask = valid_sensor_mask.unsqueeze(-1).to(attn_weights.dtype)
            mean_attn = (attn_weights * query_mask).sum(dim=1)
            mean_attn = mean_attn / query_mask.sum(dim=1).clamp_min(1.0)
        else:
            mean_attn = attn_weights.mean(dim=1)  # (B, N)
        
        # Weighted sum of values
        pooled = torch.matmul(mean_attn.unsqueeze(1), V).squeeze(1)  # (B, hidden_dim)
        
        return pooled


class DirectionHeadVMF3(nn.Module):
    """
    Direction head using von Mises-Fisher distribution on S^2 (sphere).
    Combines sensor attention pooling with global features to predict mean direction
    and concentration parameter for vmf3 distribution.
    """
    def __init__(
        self,
        sensor_feature_dim,
        upstream_feature_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        sensor_dropout=0.0,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        # Sensor processing
        self.sensor_feature_builder = SensorFeatureBuilder()
        self.sensor_encoder = SensorEncoder(
            sensor_feature_dim=sensor_feature_dim,
            hidden_dim=hidden_dim
        )
        # self.sensor_attention = SensorAttentionEncoder(
        #     hidden_dim=hidden_dim,
        #     num_heads=num_heads,
        #     num_layers=num_layers,
        #     dropout=sensor_dropout,
        # )
        self.attention_pooling = AttentionPooling(hidden_dim=hidden_dim)

        # Global processing
        self.upstream_feature_builder = UpstreamFeatureBuilder()

        # Upstream encoder
        self.upstream_encoder = UpstreamEncoder(
            upstream_feature_dim=upstream_feature_dim,
            hidden_dim=hidden_dim
        )

        # MLPs for mean direction and concentration
        self.mean_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Output 3D mean direction
        )

        self.concentration_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output concentration (kappa)
        )

    def forward(
        self,
        sensor_pos,
        agent_pos,
        agent_vel,
        self_sensor_coverage,
        remaining_sensor_coverage,
        remaining_time,
        upstream_neighbor_pos,
        upstream_neighbor_vel,
        u2s_range,
        u2u_range,
        sensor_mask=None,
        temperature=1.0,
        use_mean_direction=False,
    ):
        """
        Args:
            sensor_pos: (B, N, 3)
            agent_pos: (B, 3)
            agent_visited: (B, N)
            self_sensor_coverage: (B,) or (B, 1)
            remaining_sensor_coverage: (B,) or (B, 1)
            remaining_time: (B,) or (B, 1)
            upstream_neighbor_pos: (B, 3)
            u2s_range: scalar or (B,)
            u2u_range: scalar or (B,)
            sensor_mask: (B, N) optional
            temperature: scalar for temperature scaling

        Returns:
            dist: VonMisesFisher distribution
            mu: (B, 3) mean direction (normalized)
            kappa: (B,) concentration parameter
            direction_sample: (B, 3) sampled direction (normalized)
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        # Process sensor features
        sensor_features = self.sensor_feature_builder(
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            u2s_range=u2s_range,
        )

        sensor_embeddings = self.sensor_encoder(sensor_features)
        # contextual_embeddings = self.sensor_attention(
        #     sensor_embeddings,
        #     sensor_mask=sensor_mask,
        # )
        pooled_sensor_features = self.attention_pooling(
            sensor_embeddings,
            sensor_mask=sensor_mask,
        )

        # Process global features
        upstream_features = self.upstream_feature_builder(
            upstream_neighbor_pos=upstream_neighbor_pos,
            upstream_neighbor_vel=upstream_neighbor_vel,
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            self_sensor_coverage=self_sensor_coverage,
            remaining_sensor_coverage=remaining_sensor_coverage,
            remaining_time=remaining_time,
            u2u_range=u2u_range,
        )

        upstream_embeddings = self.upstream_encoder(upstream_features)

        # Concatenate pooled sensor features with global features
        combined_features = torch.cat(
            [pooled_sensor_features, upstream_embeddings], dim=-1
        )  # (B, hidden_dim + upstream_feature_dim)

        # Predict mean direction and concentration
        mu_raw = self.mean_mlp(combined_features)  # (B, 3)
        mu = _safe_normalize(mu_raw)  # (B, 3) normalize to unit sphere

        kappa_raw = self.concentration_mlp(combined_features)  # (B, 1)
        kappa = F.softplus(kappa_raw).squeeze(-1) + self.eps  # (B,) ensure positive

        # Apply temperature scaling to concentration
        if temperature != 1.0:
            kappa = kappa / temperature

        # VonMises3D takes eta = kappa * mu as its natural parameter
        dist = VonMises3D(kappa.unsqueeze(-1) * mu)
        direction_sample = mu if use_mean_direction else sample_vmf3(mu=mu, kappa=kappa)

        return dist, mu, kappa, direction_sample

    def log_prob(self, dist, sample):
        """Compute log probability of a sample under the distribution."""
        return dist.log_prob(sample)

    def entropy(self, kappa):
        """
        Compute entropy of vmf3 distribution using entropy_vmf3.
        """
        return entropy_vmf3(kappa=kappa)


class Actor(nn.Module):
    """
    Actor network combining sensor selection with direction sampling using vmf3.
    """
    def __init__(
        self,
        sensor_feature_dim,
        upstream_feature_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        sensor_dropout=0.0,
        direction_eps=1e-6,
    ):
        super().__init__()

        self.direction_head = DirectionHeadVMF3(
            sensor_feature_dim=sensor_feature_dim,
            upstream_feature_dim=upstream_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            sensor_dropout=sensor_dropout,
            eps=direction_eps,
        )

    def act(
        self,
        sensor_pos,
        agent_pos,
        agent_vel,
        self_sensor_coverage,
        remaining_sensor_coverage,
        remaining_time,
        upstream_neighbor_pos,
        upstream_neighbor_vel,
        u2s_range,
        u2u_range,
        sensor_mask=None,
        direction_temperature=1.0,
        use_mean_direction=False,
    ):
        """
        Sample a direction action using vmf3 distribution.

        Args:
            sensor_pos: (B, N, 3) nearby sensor positions
            agent_pos: (B, 3) agent position
            self_sensor_coverage: (B,) or (B, 1) self coverage
            remaining_sensor_coverage: (B,) or (B, 1) remaining coverage
            remaining_time: (B,) or (B, 1) remaining time
            upstream_neighbor_pos: (B, 3) upstream neighbor position
            u2s_range: scalar or (B,) UAV-to-sensor range
            u2u_range: scalar or (B,) UAV-to-UAV range
            sensor_mask: (B, N) optional mask
            direction_temperature: scalar temperature for direction sampling

        Returns:
            dict with keys:
                - direction_dist: VonMises3D distribution
                - direction_mu: (B, 3) mean direction
                - direction_kappa: (B,) concentration
                - direction_action: (B, 3) sampled direction
                - direction_log_prob: (B,) log probability of sample
        """

        direction_dist, mu, kappa, direction_sample = self.direction_head(
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            self_sensor_coverage=self_sensor_coverage,
            remaining_sensor_coverage=remaining_sensor_coverage,
            remaining_time=remaining_time,
            upstream_neighbor_pos=upstream_neighbor_pos,
            upstream_neighbor_vel=upstream_neighbor_vel,
            u2s_range=u2s_range,
            u2u_range=u2u_range,
            sensor_mask=sensor_mask,
            temperature=direction_temperature,
            use_mean_direction=use_mean_direction,
        )

        direction_log_prob = self.direction_head.log_prob(direction_dist, direction_sample)

        return {
            "direction_dist": direction_dist,
            "direction_mu": mu,
            "direction_kappa": kappa,
            "direction_action": direction_sample,
            "direction_log_prob": direction_log_prob,
            "log_prob": direction_log_prob,
        }

    def evaluate_actions(
        self,
        sensor_pos,
        agent_pos,
        agent_vel,
        self_sensor_coverage,
        remaining_sensor_coverage,
        remaining_time,
        direction_action,
        upstream_neighbor_pos,
        upstream_neighbor_vel,
        u2s_range,
        u2u_range,
        sensor_mask=None,
        direction_temperature=1.0,
        use_mean_direction=False,
    ):
        """
        Evaluate log probability and entropy of given actions.

        Args:
            (same as act)
            direction_action: (B, 3) direction action to evaluate

        Returns:
            dict with keys:
                - direction_dist: VonMisesFisher distribution
                - log_prob: (B,) log probability of actions
                - entropy: (B,) entropy of distribution
        """
        direction_dist, mu, kappa, _ = self.direction_head(
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            self_sensor_coverage=self_sensor_coverage,
            remaining_sensor_coverage=remaining_sensor_coverage,
            remaining_time=remaining_time,
            upstream_neighbor_pos=upstream_neighbor_pos,
            upstream_neighbor_vel=upstream_neighbor_vel,
            u2s_range=u2s_range,
            u2u_range=u2u_range,
            sensor_mask=sensor_mask,
            temperature=direction_temperature,
        )

        direction_log_prob = self.direction_head.log_prob(direction_dist, direction_action)
        direction_entropy = self.direction_head.entropy(kappa)

        return {
            "direction_dist": direction_dist,
            "log_prob": direction_log_prob,
            "entropy": direction_entropy,
        }
