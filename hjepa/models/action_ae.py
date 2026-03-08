from typing import Optional, NamedTuple
from torch import nn
import torch

from hjepa.configs import ConfigBase
from dataclasses import dataclass


@dataclass
class ActionAEConfig(ConfigBase):
    arch: str = "id"
    chunk_size: int = 10
    latent_dim: int = 16
    state_dim: int = 27
    action_dim: int = 8
    state_embed: str = "linear"
    state_embed_dim: int = 64
    hidden_dim: int = 256
    enc_state_cond: bool = True
    dec_state_cond: bool = True
    z_min_std: float = 0.1
    posterior_drop_p: float = 0.025
    freeze_encoder: bool = False
    chunk_on_fly: bool = True
    enc_aggregator: str = "attn"


class ActionAEResult(NamedTuple):
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    latents: torch.Tensor
    pred_actions_mean: torch.Tensor
    pred_actions_std: torch.Tensor
    pred_actions: torch.Tensor


class ActionEncoder(torch.nn.Module):
    def __init__(
        self,
        config: ActionAEConfig,
    ):
        super().__init__()
        self.config = config

        if config.enc_state_cond:
            if config.state_embed == "linear":
                self.state_embed = nn.Sequential(
                    nn.Linear(config.state_dim, config.state_embed_dim),
                    nn.ReLU(),
                )
                state_dim = config.state_embed_dim
            elif config.state_embed == "id":
                self.state_embed = nn.Identity()
                state_dim = config.state_dim
            else:
                raise ValueError(f"Unknown state_embed {config.state_embed}")
        else:
            state_dim = 0

        self.gru = nn.GRU(
            state_dim + config.action_dim,
            config.hidden_dim,
            bidirectional=True,
            batch_first=False,
        )

        if config.enc_aggregator == "attn":
            self.attention = nn.Linear(2 * config.hidden_dim, 1)

        # Mean network: 2 linear layers with ReLU after the first
        self.mean_net = nn.Sequential(
            nn.Linear(
                2 * config.hidden_dim, config.hidden_dim
            ),  # GRU is bidirectional, so output dim is 2 * hidden_dim
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        # Standard deviation network: 2 linear layers with ReLU and Softplus activation
        self.std_net = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),  # GRU is bidirectional
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Softplus(),  # Ensures positive outputs
        )

    def forward(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
    ):
        """
        Input:
            actions: (chunks, chunk_size, bs, action_dim)
            states: (chunks, chunk_size, bs, state_dim)
        Return:
            latent_mean: (chunks, bs, latent_dim),
            latent_std: (chunks, bs, latent_dim),
        """
        if self.config.enc_state_cond:
            states = self.state_embed(states)
            input_seq = torch.cat([states, actions], dim=-1)
        else:
            input_seq = actions

        chunks, chunk_size, bs, dim = input_seq.shape

        # (chunks, chunk_size, bs, _) --> (chunk_size, chunks * bs, _)
        reshaped_input_seq = input_seq.permute(1, 0, 2, 3).reshape(
            chunk_size, bs * chunks, dim
        )

        # restored_input_seq = reshaped_input_seq.reshape(chunk_size, chunks, bs, dim).permute(1, 0, 2, 3)

        hidden = self.gru(reshaped_input_seq)[0]

        # take mean over time dimension
        if self.config.enc_aggregator == "mean":
            hidden = hidden.mean(dim=0, keepdim=True)
        elif self.config.enc_aggregator == "attn":
            attention_weights = torch.softmax(self.attention(hidden), dim=0)
            hidden = (attention_weights * hidden).sum(dim=0, keepdim=True)

        # reshape into (chunks, 1, BS, dim)
        hidden = hidden.reshape(1, chunks, bs, 2 * self.config.hidden_dim).permute(
            1, 0, 2, 3
        )

        # rid of time dimension
        hidden = hidden.squeeze(1)

        latent_mean = self.mean_net(hidden)
        latent_std = self.std_net(hidden) + self.config.z_min_std

        return latent_mean, latent_std


class ActionDecoder(torch.nn.Module):
    def __init__(
        self,
        config: ActionAEConfig,
    ):
        super().__init__()
        self.config = config

        position_embed_dim = max(4, config.latent_dim // 2)

        self.position_embed = nn.Embedding(config.chunk_size, position_embed_dim)

        if self.config.dec_state_cond:
            input_dim = config.latent_dim + position_embed_dim + config.state_dim
        else:
            input_dim = config.latent_dim + position_embed_dim

        self.base_layers = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Mean network: 2 linear layers with ReLU after the first
        self.mean_net = nn.Sequential(
            nn.Linear(
                config.hidden_dim, config.hidden_dim
            ),  # GRU is bidirectional, so output dim is 2 * hidden_dim
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

        # Standard deviation network: 2 linear layers with ReLU and Softplus activation
        self.std_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),  # GRU is bidirectional
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Softplus(),  # Ensures positive outputs
        )

    def forward_single_t(
        self,
        states: torch.Tensor,
        latents: torch.Tensor,
        timestep: int,
    ):
        """
        Input:
            states: (bs, state_dim)
            latents: (bs, latent_dim)
        Return:
            actions: (bs, action_dim)
        """

        timestep = torch.tensor([timestep], device=states.device)
        pos_embeddings = self.position_embed(timestep)  # Shape: (1, position_embed_dim)
        pos_embeddings = pos_embeddings.expand(
            states.size(0), -1
        )  # Expand to match batch size

        if self.config.dec_state_cond:
            hidden = torch.cat([states, latents, pos_embeddings], dim=-1)
        else:
            hidden = torch.cat([latents, pos_embeddings], dim=-1)

        hidden = self.base_layers(hidden)

        action_mean = self.mean_net(hidden)
        action_std = self.std_net(hidden)
        pred_actions = torch.randn_like(action_mean) * action_std + action_mean

        return pred_actions, action_mean, action_std

    def forward(
        self,
        states: torch.Tensor,
        latents: torch.Tensor,
    ):
        """
        Input:
            states: (chunks, chunk_size, bs, state_dim)
            latents: (chunks, chunk_size, bs, latent_dim)
        Return:
            actions: (chunks, chunk_size, bs, action_dim)
        """
        # Generate positional indices for each timestep in the chunk
        chunk_size = states.size(
            1
        )  # Assuming shape is (chunks, chunk_size, bs, state_dim)
        device = states.device  # Ensure the positional indices are on the same device
        t_indices = torch.arange(chunk_size, device=device).expand(
            states.size(0), -1
        )  # Shape: (chunks, chunk_size)

        # Get positional embeddings
        pos_embeddings = self.position_embed(
            t_indices
        )  # Shape: (chunks, chunk_size, position_embed_dim)
        pos_embeddings = pos_embeddings.unsqueeze(2).expand(
            -1, -1, states.size(2), -1
        )  # Expand to match batch size
        # Shape: (chunks, chunk_size, bs, position_embed_dim)

        if self.config.dec_state_cond:
            hidden = torch.cat([states, latents, pos_embeddings], dim=-1)
        else:
            hidden = torch.cat([latents, pos_embeddings], dim=-1)

        hidden = self.base_layers(hidden)

        action_mean = self.mean_net(hidden)
        action_std = self.std_net(hidden)
        pred_actions = torch.randn_like(action_mean) * action_std + action_mean

        return pred_actions, action_mean, action_std


class ActionAE(torch.nn.Module):
    def __init__(
        self,
        config: ActionAEConfig,
    ):
        super().__init__()
        self.config = config
        self.encoder = ActionEncoder(
            config=config,
        )
        self.decoder = ActionDecoder(
            config=config,
        )

    def chunk_seq(self, x, skip_last=False):
        """
        Input:
            x: (T, bs, dim)
        Output:
            x: (chunks, chunk_size, bs, dim)
        """

        T, bs, dim = x.shape

        new_shape = (
            T // self.config.chunk_size,
            self.config.chunk_size,
            bs,
            dim,
        )
        if skip_last:
            assert not (T - 1) % self.config.chunk_size
            chunked_x = x[:-1].view(*new_shape[:-1], -1)
        else:
            assert not T % self.config.chunk_size
            chunked_x = x.view(new_shape)

        return chunked_x

    def forward(
        self,
        actions: torch.Tensor,
        proprio_states: Optional[torch.Tensor] = None,
    ):
        """
        Input:
            actions: (T, bs, action_dim) OR (chunks, bs, chunk_size, action_dim)
            proprio_states: (T+1, bs, qpos_dim) OR (chunks + 1, bs, chunk_size, qpos_dim)
        Output:
            latent_mean: (chunks, bs, latent_dim),
            latent_std: (chunks, bs, latent_dim),
            pred_actions: (T, bs, action_dim) OR (chunks, bs, chunk_size, action_dim)
        """
        if self.config.chunk_on_fly:
            T, bs, action_dim = actions.shape
            assert not T % self.config.chunk_size
            # divide into chunks along temporal dimension
            chunked_actions = self.chunk_seq(actions)
            chunked_states = self.chunk_seq(proprio_states, skip_last=True)
        else:
            chunks, bs, chunk_size, action_dim = actions.shape
            assert chunk_size == self.config.chunk_size
            assert proprio_states.shape[2] == self.config.chunk_size
            # chunks, bs, chunk_size, dim --> chunks, chunk_size * bs, dim
            chunked_actions = actions.permute(0, 2, 1, 3)
            chunked_states = proprio_states[:-1].permute(0, 2, 1, 3)

        encoder_context = (
            torch.no_grad if self.config.freeze_encoder else torch.enable_grad
        )

        with encoder_context():
            latent_mean, latent_std = self.encoder(
                actions=chunked_actions,
                states=chunked_states,
            )

        # sample latents
        latents = torch.randn_like(latent_mean) * latent_std + latent_mean

        # posterior drop out. here we assume we have uniform (0,1) prior
        if self.training:
            mask = torch.rand_like(latents) < self.config.posterior_drop_p
            priors = torch.normal(mean=0, std=1, size=latents.shape).to(latents.device)
            latents[mask] = priors[mask]

        repeated_latents = latents.unsqueeze(1).repeat(
            1, chunked_actions.shape[1], 1, 1
        )

        pred_actions, action_mean, action_std = self.decoder(
            states=chunked_states,
            # repeat latents along chunk_size dimension
            latents=repeated_latents,
        )

        if self.config.chunk_on_fly:
            action_mean = action_mean.view(T, bs, action_dim)
            action_std = action_std.view(T, bs, action_dim)
            pred_actions = pred_actions.view(T, bs, action_dim)
        else:
            action_mean = action_mean.permute(0, 2, 1, 3)
            action_std = action_std.permute(0, 2, 1, 3)
            pred_actions = pred_actions.permute(0, 2, 1, 3)

        result = ActionAEResult(
            latent_mean=latent_mean,
            latent_std=latent_std,
            latents=latents,
            pred_actions_mean=action_mean,
            pred_actions_std=action_std,
            pred_actions=pred_actions,
        )

        return result


def build_action_ae(
    config: ActionAEConfig,
):
    if config.arch == "canonical":
        return ActionAE(
            config=config,
        )
    elif config.arch == "id":
        return None
    else:
        raise ValueError(f"Unknown arch {config.arch}")
