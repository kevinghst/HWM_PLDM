from dataclasses import dataclass, field
import enum
from typing import List

from hjepa.objectives.laplacian import (
    LaplacianObjective,
    LaplacianObjectiveConfig,
)  # noqa
from hjepa.objectives.vicreg import VICRegObjective, VICRegObjectiveConfig  # noqa
from hjepa.objectives.idm import IDMObjective, IDMObjectiveConfig  # noqa
from hjepa.objectives.reconstruction import (
    ReconstructionObjective,
    ReconstructionObjectiveConfig,
)
from hjepa.objectives.action_ae import AAEObjective, AAEObjectiveConfig
from hjepa.objectives.aae_probe import AAEProbeObjective, AAEProbeConfig
from hjepa.objectives.aae_probe_chunk import AAEProbeChunkObjective, AAEProbeChunkConfig
from hjepa.objectives.kl import KLObjective, KLObjectiveConfig
from hjepa.objectives.prediction import PredictionObjective, PredictionObjectiveConfig
from hjepa.objectives.probe import ProbeObjective, ProbeObjectiveConfig


class ObjectiveType(enum.Enum):
    VICReg = enum.auto()
    VICRegObs = enum.auto()
    VICRegProprio = enum.auto()
    VICRegLocation = enum.auto()
    RSSM = enum.auto()
    SimCLR = enum.auto()
    HJEPA = enum.auto()
    Reconstruction = enum.auto()
    IDM = enum.auto()
    Laplacian = enum.auto()
    KL = enum.auto()
    Prediction = enum.auto()
    PredictionObs = enum.auto()
    PredictionProprio = enum.auto()
    PredictionRawLocation = enum.auto()
    ProbeLocation = enum.auto()
    ProbeProprioVel = enum.auto()
    ActionAE = enum.auto()
    AAEProbe = enum.auto()
    AAEProbeChunk = enum.auto()


@dataclass
class ObjectivesConfig:
    objectives: List[ObjectiveType] = field(default_factory=lambda: [])
    vicreg: VICRegObjectiveConfig = VICRegObjectiveConfig()
    vicreg_obs: VICRegObjectiveConfig = VICRegObjectiveConfig()
    vicreg_proprio: VICRegObjectiveConfig = VICRegObjectiveConfig()
    vicreg_location: VICRegObjectiveConfig = VICRegObjectiveConfig()
    idm: IDMObjectiveConfig = IDMObjectiveConfig()
    laplacian: LaplacianObjectiveConfig = LaplacianObjectiveConfig()
    reconstruction: ReconstructionObjectiveConfig = ReconstructionObjectiveConfig()
    kl: KLObjectiveConfig = KLObjectiveConfig()
    prediction: PredictionObjectiveConfig = PredictionObjectiveConfig()
    prediction_obs: PredictionObjectiveConfig = PredictionObjectiveConfig()
    prediction_proprio: PredictionObjectiveConfig = PredictionObjectiveConfig()
    prediction_raw_location: PredictionObjectiveConfig = PredictionObjectiveConfig()
    probe: ProbeObjectiveConfig = ProbeObjectiveConfig()
    action_ae: AAEObjectiveConfig = AAEObjectiveConfig()
    aae_probe: AAEProbeConfig = AAEProbeConfig()
    aae_probe_chunk: AAEProbeChunkConfig = AAEProbeChunkConfig()

    def build_objectives_list(
        self,
        repr_dim: int,
        name_prefix: str = "",
    ):
        objectives = []
        for objective_type in self.objectives:
            if objective_type == ObjectiveType.VICReg:
                objectives.append(
                    VICRegObjective(
                        self.vicreg, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.VICRegObs:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_obs,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.VICRegProprio:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_proprio,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="proprio",
                    )
                )
            elif objective_type == ObjectiveType.VICRegLocation:
                objectives.append(
                    VICRegObjective(
                        self.vicreg_location,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_attr="locations",
                    )
                )
            elif objective_type == ObjectiveType.IDM:
                objectives.append(
                    IDMObjective(self.idm, name_prefix=name_prefix, repr_dim=repr_dim)
                )
            elif objective_type == ObjectiveType.Laplacian:
                objectives.append(
                    LaplacianObjective(
                        self.laplacian, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.Reconstruction:
                objectives.append(
                    ReconstructionObjective(
                        self.reconstruction, name_prefix=name_prefix, repr_dim=repr_dim
                    )
                )
            elif objective_type == ObjectiveType.KL:
                objectives.append(KLObjective(self.kl, name_prefix=name_prefix))
            elif objective_type == ObjectiveType.Prediction:
                objectives.append(
                    PredictionObjective(
                        self.prediction,
                        name_prefix=name_prefix,
                        pred_attr="state",
                    )
                )
            elif objective_type == ObjectiveType.PredictionObs:
                objectives.append(
                    PredictionObjective(
                        self.prediction_obs,
                        name_prefix=name_prefix,
                        pred_attr="obs",
                    )
                )
            elif objective_type == ObjectiveType.PredictionProprio:
                objectives.append(
                    PredictionObjective(
                        self.prediction_proprio,
                        name_prefix=name_prefix,
                        pred_attr="proprio",
                    )
                )
            elif objective_type == ObjectiveType.PredictionRawLocation:
                objectives.append(
                    PredictionObjective(
                        self.prediction_raw_location,
                        name_prefix=name_prefix,
                        pred_attr="raw_locations",
                    )
                )
            elif objective_type == ObjectiveType.ProbeLocation:
                objectives.append(
                    ProbeObjective(
                        self.probe,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_dim=2,
                        probe_target="locations",
                    )
                )
            elif objective_type == ObjectiveType.ProbeProprioVel:
                objectives.append(
                    ProbeObjective(
                        self.probe,
                        name_prefix=name_prefix,
                        repr_dim=repr_dim,
                        pred_dim=2,
                        probe_target="proprio_vel",
                    )
                )
            elif objective_type == ObjectiveType.ActionAE:
                objectives.append(
                    AAEObjective(
                        self.action_ae,
                        name_prefix=name_prefix,
                    )
                )
            elif objective_type == ObjectiveType.AAEProbe:
                objectives.append(
                    AAEProbeObjective(
                        self.aae_probe,
                        name_prefix=name_prefix,
                    )
                )
            elif objective_type == ObjectiveType.AAEProbeChunk:
                objectives.append(
                    AAEProbeChunkObjective(
                        self.aae_probe_chunk,
                        name_prefix=name_prefix,
                    )
                )
            else:
                raise NotImplementedError()
        return objectives
