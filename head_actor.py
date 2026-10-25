# Copyright © MERL 2026
# Description: Defines the head UAV actor network for sensor selection and direction control.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pyro.distributions import VonMises3D
from vmf3_sampler import sample_vmf3, entropy_vmf3



class StateFeatureBuilder(nn.Module):
    def __init__(self, pos_dim):
        super().__init__()
        self.pos_dim = pos_dim

    def forward(self, sensor_pos, energy_to_reach, time_to_reach, visited, global_state):
        """
        sensor_pos:      (B, N, d)
        energy_to_reach: (B, N)
        time_to_reach:   (B, N)
        visited:         (B, N)
        global_state:    (B, G)
        """
        agent_pos = global_state[:, : self.pos_dim]
        n_sensors = sensor_pos.shape[1]

        rel_pos = sensor_pos - agent_pos.unsqueeze(1)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)

        visited = visited.unsqueeze(-1).float()
        energy_to_reach = energy_to_reach.unsqueeze(-1)
        time_to_reach = time_to_reach.unsqueeze(-1)

        remaining_energy = global_state[:, self.pos_dim : self.pos_dim + 1].unsqueeze(1).expand(-1, n_sensors, -1)
        remaining_time = global_state[:, self.pos_dim + 1 : self.pos_dim + 2].unsqueeze(1).expand(-1, n_sensors, -1)
        collected_sensors = global_state[:, self.pos_dim + 2 : self.pos_dim + 3].unsqueeze(1).expand(-1, n_sensors, -1)
        remaining_sensors = global_state[:, self.pos_dim + 3 : self.pos_dim + 4].unsqueeze(1).expand(-1, n_sensors, -1)
        altitude_buffer = global_state[:, self.pos_dim + 4 : self.pos_dim + 5].unsqueeze(1).expand(-1, n_sensors, -1)

        return torch.cat(
            [
                sensor_pos,
                rel_pos,
                dist,
                energy_to_reach,
                time_to_reach,
                visited,
                remaining_energy,
                remaining_time,
                collected_sensors,
                remaining_sensors,
                altitude_buffer,
            ],
            dim=-1,
        )


class StateEncoder(nn.Module):
    def __init__(self, state_feature_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state_features):
        return self.mlp(state_features)


class StateAttentionEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, state_embeddings, sensor_mask=None):
        padding_mask = None if sensor_mask is None else ~sensor_mask.bool()
        return self.transformer(state_embeddings, src_key_padding_mask=padding_mask)


class SensorSelectionHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, contextual_state_embeddings, visited, sensor_mask=None, temperature=1.0):
        logits = self.score_mlp(contextual_state_embeddings).squeeze(-1)
        logits = logits / temperature

        invalid = visited.bool()
        if sensor_mask is not None:
            invalid = invalid | (~sensor_mask.bool())

        logits = logits.masked_fill(invalid, -1e9)
        return logits


class DirectionHead(nn.Module):
    def __init__(self, pos_dim, hidden_dim=128):
        super().__init__()
        self.kappa_head = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.kappa_head[-1].bias, 1.5)

    def forward(
        self,
        sensor_action,
        sensor_pos,
        agent_pos,
        energy_to_reach,
        time_to_reach,
        remaining_energy,
        remaining_time,
    ):
        b = sensor_action.shape[0]
        idx = torch.arange(b, device=sensor_action.device)

        chosen_pos = sensor_pos[idx, sensor_action]
        rel_vec = chosen_pos - agent_pos
        mu = F.normalize(rel_vec, p=2, dim=-1)

        e2r = energy_to_reach[idx, sensor_action].unsqueeze(-1)
        t2r = time_to_reach[idx, sensor_action].unsqueeze(-1)
        re = remaining_energy.unsqueeze(-1)
        rt = remaining_time.unsqueeze(-1)
        kappa_input = torch.cat([e2r, t2r, re, rt], dim=-1)
        kappa_input = torch.clamp(kappa_input, 0.0, 5.0)

        kappa = self.kappa_head(kappa_input)
        kappa = F.softplus(kappa) + 1e-6
        kappa = torch.clamp(kappa, 0.1, 100.0)

        direction_dist = VonMises3D(mu * kappa)
        return direction_dist, kappa, mu


class Actor(nn.Module):
    """
    Head actor borrowed from single-agent PPO architecture.

    Exposes MAPPO-friendly methods:
    - act(...): returns dict with sampled actions/log-probs
    - evaluate_actions(...): returns dict with log_prob and entropy
    """

    def __init__(
        self,
        pos_dim,
        state_feature_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        self.pos_dim = pos_dim
        self.feature_builder = StateFeatureBuilder(pos_dim)
        self.state_encoder = StateEncoder(state_feature_dim=state_feature_dim, hidden_dim=hidden_dim)
        self.state_attention_encoder = StateAttentionEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.sensor_selector = SensorSelectionHead(hidden_dim=hidden_dim)
        self.direction_head = DirectionHead(pos_dim=pos_dim, hidden_dim=hidden_dim)

    def encode(self, sensor_pos, energy_to_reach, time_to_reach, visited, global_state, sensor_mask=None):
        state_features = self.feature_builder(
            sensor_pos=sensor_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            visited=visited,
            global_state=global_state,
        )
        state_embeddings = self.state_encoder(state_features)
        return self.state_attention_encoder(state_embeddings, sensor_mask=sensor_mask)

    def _forward_sensor_dist(
        self,
        sensor_pos,
        energy_to_reach,
        time_to_reach,
        visited,
        global_state,
        sensor_mask=None,
        temperature=1.0,
    ):
        contextual_embeddings = self.encode(
            sensor_pos=sensor_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            visited=visited,
            global_state=global_state,
            sensor_mask=sensor_mask,
        )
        logits = self.sensor_selector(
            contextual_state_embeddings=contextual_embeddings,
            visited=visited,
            sensor_mask=sensor_mask,
            temperature=temperature,
        )
        return Categorical(logits=logits)

    @staticmethod
    def _safe_entropy(direction_dist, sensor_entropy):
        try:
            direction_entropy = direction_dist.entropy()
            if direction_entropy.ndim > 1:
                direction_entropy = direction_entropy.sum(dim=-1)
            return sensor_entropy + direction_entropy
        except Exception:
            return sensor_entropy

    def get_direction(
        self,
        sensor_action,
        sensor_pos,
        agent_pos,
        energy_to_reach,
        time_to_reach,
        remaining_energy,
        remaining_time,
    ):
        return self.direction_head(
            sensor_action=sensor_action,
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            remaining_energy=remaining_energy,
            remaining_time=remaining_time,
        )

    def act(
        self,
        sensor_pos,
        energy_to_reach,
        time_to_reach,
        visited,
        global_state,
        sensor_mask=None,
        sensor_temperature=1.0,
        direction_temperature=1.0,
    ):
        del direction_temperature  # Not used for VonMises head in this architecture.

        sensor_dist = self._forward_sensor_dist(
            sensor_pos=sensor_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            visited=visited,
            global_state=global_state,
            sensor_mask=sensor_mask,
            temperature=sensor_temperature,
        )

        sensor_action = sensor_dist.sample()
        sensor_log_prob = sensor_dist.log_prob(sensor_action)

        agent_pos = global_state[:, : self.pos_dim]
        remaining_energy = global_state[:, self.pos_dim]
        remaining_time = global_state[:, self.pos_dim + 1]

        direction_dist, kappa, mu = self.get_direction(
            sensor_action=sensor_action,
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            remaining_energy=remaining_energy,
            remaining_time=remaining_time,
        )

        direction_action = sample_vmf3(mu, kappa)
        direction_log_prob = direction_dist.log_prob(direction_action)

        return {
            "sensor_dist": sensor_dist,
            "sensor_action": sensor_action,
            "sensor_log_prob": sensor_log_prob,
            "direction_dist": direction_dist,
            "direction_action": direction_action,
            "direction_weights": direction_action,
            "direction_log_prob": direction_log_prob,
            "kappa": kappa,
            "mu": mu,
            "log_prob": sensor_log_prob + direction_log_prob,
        }

    def evaluate_actions(
        self,
        sensor_pos,
        energy_to_reach,
        time_to_reach,
        visited,
        global_state,
        sensor_action,
        direction_action=None,
        direction_weights=None,
        sensor_mask=None,
        sensor_temperature=1.0,
        direction_temperature=1.0,
    ):
        del direction_temperature  # Not used for VonMises head in this architecture.

        if direction_action is None and direction_weights is None:
            raise ValueError("Either direction_action or direction_weights must be provided.")
        if direction_action is None:
            direction_action = direction_weights

        sensor_dist = self._forward_sensor_dist(
            sensor_pos=sensor_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            visited=visited,
            global_state=global_state,
            sensor_mask=sensor_mask,
            temperature=sensor_temperature,
        )

        sensor_log_prob = sensor_dist.log_prob(sensor_action)
        sensor_entropy = sensor_dist.entropy()

        agent_pos = global_state[:, : self.pos_dim]
        remaining_energy = global_state[:, self.pos_dim]
        remaining_time = global_state[:, self.pos_dim + 1]

        direction_dist, _, _ = self.get_direction(
            sensor_action=sensor_action,
            sensor_pos=sensor_pos,
            agent_pos=agent_pos,
            energy_to_reach=energy_to_reach,
            time_to_reach=time_to_reach,
            remaining_energy=remaining_energy,
            remaining_time=remaining_time,
        )
        direction_log_prob = direction_dist.log_prob(direction_action)

        total_entropy = self._safe_entropy(direction_dist, sensor_entropy)

        return {
            "sensor_dist": sensor_dist,
            "direction_dist": direction_dist,
            "log_prob": sensor_log_prob + direction_log_prob,
            "entropy": total_entropy,
        }
