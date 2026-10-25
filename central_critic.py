# Copyright © MERL 2026
# Description: Defines centralized critic encoders and value estimation for MAPPO.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SensorAttentionPooling(nn.Module):
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
            scores = scores.masked_fill(
                ~valid_sensor_mask.unsqueeze(1),  # (B, 1, N)
                -1e9,
            )
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, N, N)
        
        if sensor_mask is not None:
            # Exclude padded sensors as both keys and queries. Renormalizing after
            # masking also makes an all-invalid row produce zero attention.
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


class FollowerSensorFeatureBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        sensor_pos,
        agent_pos,
        u2s_range,
    ):
        """
        sensor_pos:  (Bf, N, d)
        agent_pos:           (Bf, d)
        u2s_range:           scalar or (Bf, 1)
        visited_own:         (Bf, N)
        visited_upstream:    (Bf, M)
        """
        # combined_sensor_pos = sensor_pos
        rel_sensor_pos = sensor_pos - agent_pos.unsqueeze(1)
        rel_sensor_dir = F.normalize(rel_sensor_pos, p=2, dim=-1, eps=1e-8)
    
        u2s = torch.as_tensor(u2s_range, device=sensor_pos.device, dtype=sensor_pos.dtype)
        if u2s.ndim == 1:
            u2s = u2s.view(-1, 1, 1)
        elif u2s.ndim == 2:
            u2s = u2s.unsqueeze(1)

        sensor_dist = torch.norm(rel_sensor_pos, dim=-1, keepdim=True) - u2s

        return torch.cat(
            [
                agent_pos.unsqueeze(1).expand(-1, sensor_pos.shape[1], -1),
                rel_sensor_dir,
                sensor_dist,
            ],
            dim=-1,
        )


class UpstreamFeatureBuilder(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _expand_scalar_feature(value, batch_size, device, dtype):
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.view(1, 1)
        elif value.ndim == 1:
            value = value.view(-1, 1)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        return value

    @staticmethod
    def _safe_normalize(vec, eps=1e-8):
        return F.normalize(vec, p=2, dim=-1, eps=eps)

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
        """
        Builds direction-head style upstream features, excluding selected-sensor terms.

        upstream_neighbor_pos: (Bf, d)
        agent_pos:             (Bf, d)
        self_sensor_coverage:  (Bf, 1)
        remaining_sensor_coverage: (Bf, 1)
        remaining_time:        (Bf, 1)
        u2u_range:             scalar or (Bf, 1)
        """
        upstream_rel = upstream_neighbor_pos - agent_pos
        upstream_dist = torch.norm(upstream_rel, dim=-1, keepdim=True)
        upstream_dir = self._safe_normalize(upstream_rel)

        b = agent_pos.shape[0]
        u2u_range = self._expand_scalar_feature(u2u_range, b, agent_pos.device, agent_pos.dtype)

        upstream_link_margin = u2u_range - upstream_dist
        upstream_rel_vel = upstream_neighbor_vel - agent_vel
        upstream_rel_normalized_vel = self._safe_normalize(upstream_rel_vel)

        return torch.cat(
            [
                agent_pos,
                upstream_dir,
                upstream_dist,
                upstream_rel_normalized_vel,
                self_sensor_coverage.float().view(-1, 1),
                remaining_sensor_coverage.float().view(-1, 1),
                remaining_time.float().view(-1, 1),
                upstream_link_margin,
            ],
            dim=-1,
        )


class CentralCritic(nn.Module):
    """
    Simple centralized critic without graph message passing.

    Head embedding: same masked-mean contextual embedding logic as the single-agent critic input.
    Follower embedding:
      1) contextual sensor embeddings from follower sensor stack,
      2) upstream-neighbor embedding from direction-head style upstream-only features,
      3) concatenate upstream embedding to each contextual sensor embedding,
      4) pool across sensors, then across followers.
    """

    def __init__(
        self,
        follower_sensor_feature_dim,
        upstream_feature_dim,
        hidden_dim=128,
        global_feature_dim=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_feature_dim = global_feature_dim

        self.follower_sensor_feature_builder = FollowerSensorFeatureBuilder()
        self.follower_sensor_encoder = nn.Sequential(
            nn.Linear(follower_sensor_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sensor_attention_pooling = SensorAttentionPooling(hidden_dim=hidden_dim)

        self.upstream_feature_builder = UpstreamFeatureBuilder()
        self.upstream_encoder = nn.Sequential(
            nn.Linear(upstream_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.follower_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        value_in_dim = hidden_dim + (global_feature_dim if global_feature_dim > 0 else 0)
        self.value_head = nn.Sequential(
            nn.Linear(value_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(x, mask, dim):
        if mask is None:
            return x.mean(dim=dim)
        w = mask.unsqueeze(-1).float()
        denom = w.sum(dim=dim).clamp_min(1.0)
        return (x * w).sum(dim=dim) / denom

    def compute_follower_embedding(
        self,
        follower_sensor_pos,
        follower_pos,
        follower_self_sensor_coverage,
        follower_remaining_sensor_coverage,
        follower_remaining_time,
        upstream_neighbor_pos,
        u2s_range,
        u2u_range,
        follower_vel=None,
        upstream_neighbor_vel=None,
        follower_sensor_mask=None,
        follower_mask=None,
    ):
        """
        follower_sensor_pos:   (B, F, N, d)
        upstream_sensor_pos:   (B, F, M, d), optional
        follower_pos:          (B, F, d)
        follower_visited:      (B, F, N)
        upstream_visited:      (B, F, M), optional
        follower_self_sensor_coverage: (B, F, 1)
        follower_remaining_sensor_coverage: (B, F, 1)
        follower_remaining_time: (B, F, 1)
        upstream_neighbor_pos: (B, F, d)
        u2u_range:             scalar or (B, F) or (B, F, 1)
        follower_sensor_mask:  (B, F, N), optional
        follower_upstream_sensor_mask: (B, F, M), optional
        follower_mask:         (B, F), optional
        """
        b, f, n, d = follower_sensor_pos.shape
        # n_upstream = int(upstream_sensor_pos.shape[2]) if upstream_sensor_pos is not None else 0

        if follower_vel is None:
            follower_vel = torch.zeros_like(follower_pos)
        if upstream_neighbor_vel is None:
            upstream_neighbor_vel = torch.zeros_like(upstream_neighbor_pos)

        flat_sensor_pos = follower_sensor_pos.reshape(b * f, n, d)
        flat_pos = follower_pos.reshape(b * f, d)
        flat_vel = follower_vel.reshape(b * f, d)
        flat_self_sensor_coverage = follower_self_sensor_coverage.reshape(b * f, 1)
        flat_remaining_sensor_coverage = follower_remaining_sensor_coverage.reshape(b * f, 1)
        flat_remaining_time = follower_remaining_time.reshape(b * f, 1)
        flat_upstream = upstream_neighbor_pos.reshape(b * f, d)
        flat_upstream_vel = upstream_neighbor_vel.reshape(b * f, d)

        if follower_sensor_mask is not None:
            flat_sensor_mask = follower_sensor_mask.reshape(b * f, n)
        else:
            flat_sensor_mask = None

        if torch.is_tensor(u2s_range):
            if u2s_range.ndim == 3:
                flat_u2s_range = u2s_range.reshape(b * f, 1)
            elif u2s_range.ndim == 2:
                flat_u2s_range = u2s_range.reshape(b * f, 1)
            elif u2s_range.ndim == 1 and u2s_range.shape[0] == b:
                flat_u2s_range = u2s_range.unsqueeze(1).repeat(1, f).reshape(b * f, 1)
            else:
                flat_u2s_range = u2s_range
        else:
            flat_u2s_range = u2s_range

        if torch.is_tensor(u2u_range):
            if u2u_range.ndim == 3:
                flat_u2u_range = u2u_range.reshape(b * f, 1)
            elif u2u_range.ndim == 2:
                flat_u2u_range = u2u_range.reshape(b * f, 1)
            elif u2u_range.ndim == 1 and u2u_range.shape[0] == b:
                flat_u2u_range = u2u_range.unsqueeze(1).repeat(1, f).reshape(b * f, 1)
            else:
                flat_u2u_range = u2u_range
        else:
            flat_u2u_range = u2u_range

        sensor_features = self.follower_sensor_feature_builder(
            sensor_pos=flat_sensor_pos,
            agent_pos=flat_pos,
            u2s_range=flat_u2s_range,
        )
    
        sensor_embeddings = self.follower_sensor_encoder(sensor_features)
        pooled_sensor_features = self.sensor_attention_pooling(
            sensor_embeddings,
            sensor_mask=flat_sensor_mask,
        )

        upstream_features = self.upstream_feature_builder(
            upstream_neighbor_pos=flat_upstream,
            upstream_neighbor_vel=flat_upstream_vel,
            agent_pos=flat_pos,
            agent_vel=flat_vel,
            self_sensor_coverage=flat_self_sensor_coverage,
            remaining_sensor_coverage=flat_remaining_sensor_coverage,
            remaining_time=flat_remaining_time,
            u2u_range=flat_u2u_range,
        )
        upstream_embedding = self.upstream_encoder(upstream_features)

        combined_features = torch.cat(
            [pooled_sensor_features, upstream_embedding], dim=-1
        )

        per_follower_embedding = self.follower_fusion(combined_features).view(b, f, -1)

        if follower_mask is not None:
            follower_embedding = self._masked_mean(per_follower_embedding, follower_mask, dim=1)
        else:
            follower_embedding = per_follower_embedding.mean(dim=1)

        return follower_embedding


    def forward(
        self,
        follower_sensor_pos,
        follower_pos,
        follower_self_sensor_coverage,
        follower_remaining_sensor_coverage,
        follower_remaining_time,
        upstream_neighbor_pos,
        u2s_range,
        u2u_range,
        follower_vel=None,
        upstream_neighbor_vel=None,
        follower_sensor_mask=None,
        follower_mask=None,
        global_features=None,
        **kwargs,  # absorbs unused head_* keys from critic_obs
    ):
        follower_embedding = self.compute_follower_embedding(
            follower_sensor_pos=follower_sensor_pos,
            follower_pos=follower_pos,
            follower_self_sensor_coverage=follower_self_sensor_coverage,
            follower_remaining_sensor_coverage=follower_remaining_sensor_coverage,
            follower_remaining_time=follower_remaining_time,
            upstream_neighbor_pos=upstream_neighbor_pos,
            u2s_range=u2s_range,
            u2u_range=u2u_range,
            follower_vel=follower_vel,
            upstream_neighbor_vel=upstream_neighbor_vel,
            follower_sensor_mask=follower_sensor_mask,
            follower_mask=follower_mask,
        )

        critic_input = follower_embedding

        if self.global_feature_dim > 0:
            if global_features is None:
                raise ValueError("global_features must be provided when global_feature_dim > 0")
            critic_input = torch.cat([critic_input, global_features], dim=-1)

        value = self.value_head(critic_input).squeeze(-1)
        return value
