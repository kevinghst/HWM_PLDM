from pldm.configs import ConfigBase
from dataclasses import dataclass
from typing import Optional, NamedTuple
import torch


@dataclass
class PredictorConfig(ConfigBase):
    use_vmap: bool = False
    predictor_arch: str = "rnnV3"
    predictor_subclass: str = "a"
    predictor_ln: bool = False
    rnn_state_dim: int = 512
    rnn_converter_arch: str = ""
    z_discrete: bool = False
    z_discrete_dim: int = 16
    z_discrete_dists: int = 16
    z_dim: int = 0
    z_min_std: float = 0.1
    z_stochastic: bool = True
    posterior_drop_p: float = 0.0
    prior_arch: str = "512"
    posterior_arch: str = "512"
    posterior_input_type: str = "term_states"
    posterior_input_dim: Optional[int] = None
    action_encoder_arch: str = ""
    residual: bool = False
    rnn_layers: int = 1
    tie_backbone_ln: bool = False
    dropout: float = 0.0
    ensemble_size: int = 1  # when set to 1, there is no ensemble.
    prefused_repr: bool = True
    transformer_num_layers: Optional[int] = None
    transformer_activation: Optional[str] = None
    transformer_nhead: Optional[int] = None
    transformer_dim_feedforward: Optional[int] = None
    transformer_teacher_forcing_ratio: Optional[float] = None
    transformer_max_seq_len: Optional[int] = None
    use_teacher_forcing: Optional[bool] = None
    use_checkpointing: Optional[bool] = None


class PredictorOutput(NamedTuple):
    predictions: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    proprio_component: Optional[torch.Tensor] = None
    location_component: Optional[torch.Tensor] = None
    raw_locations: Optional[torch.Tensor] = None
    ensemble_predictions: Optional[torch.Tensor] = None
    ensemble_obs_component: Optional[torch.Tensor] = None
    ensemble_proprio_component: Optional[torch.Tensor] = None
    ensemble_location_component: Optional[torch.Tensor] = None
    ensemble_raw_locations: Optional[torch.Tensor] = None
    prior_mus: Optional[torch.Tensor] = None
    prior_vars: Optional[torch.Tensor] = None
    prior_logits: Optional[torch.Tensor] = None
    priors: Optional[torch.Tensor] = None
    posterior_mus: Optional[torch.Tensor] = None
    posterior_vars: Optional[torch.Tensor] = None
    posterior_logits: Optional[torch.Tensor] = None
    posteriors: Optional[torch.Tensor] = None


class SingleStepPredictorOutput(NamedTuple):
    prediction: torch.Tensor
    obs_component: Optional[torch.Tensor] = None
    proprio_component: Optional[torch.Tensor] = None
    next_hidden_state: Optional[torch.Tensor] = None  # for RNNs
    location_component: Optional[torch.Tensor] = None
    raw_location: Optional[torch.Tensor] = None
