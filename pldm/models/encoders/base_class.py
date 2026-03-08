from torch import nn
from pldm.models.encoders.enums import BackboneOutput


class SequenceBackbone(nn.Module):
    def __init__(self):
        """
        collapse T and BS dimensions prior to passing to backbone
        afterwards reshape to original shape
        """
        super().__init__()
        self.output_proprio_dim = 0
        self.using_location = False
        self.using_proprio = False

    def _remove_proprio_component_for_spatial(self, embeddings):
        """
        remove the proprio component from spatial embeddings

        Input:
            embeddings: tensor
            (T, BS, Ch, W, H) or
            (BS, Ch, W, H) or
            (T, BS, H) or
            (BS, H)
        """
        og_shape = tuple(embeddings.shape)
        flattened_input = len(og_shape) < 4

        # first reshape to spatial dimension if needed
        if flattened_input:
            spatial_shape = (*embeddings.shape[:-1], *self.output_dim)
            embeddings = embeddings.view(spatial_shape)

        proprio_channels = self.output_proprio_dim[0]

        # remove the proprio dimensions
        if len(embeddings.shape) == 5:
            embeddings = embeddings[:, :, :-proprio_channels]
        elif len(embeddings.shape) == 4:
            embeddings = embeddings[:, :-proprio_channels]

        # reflatten tensor if needed
        if flattened_input:
            embeddings = embeddings.view(*og_shape[:-1], -1)

        return embeddings

    def remove_proprio_component(self, embeddings):
        """
        remove the proprio component from embeddings
        Input:
            embeddings: tensor
            (T, BS, Ch, W, H) or
            (BS, Ch, W, H) or
            (T, BS, H) or
            (BS, H)
        """
        if not self.output_proprio_dim:
            return embeddings

        if isinstance(self.output_dim, int):
            return embeddings[..., : -self.output_proprio_dim]
        else:
            return self._remove_proprio_component_for_spatial(embeddings)

    def forward_multiple(self, x, proprio=None, locations=None):
        """
        input:
            x: [T, BS, *] or [BS, *]
        output:
            x: [T, BS, *] or [BS, *]
        """

        # if no time dimension, just feed it directly to backbone
        if x.dim() == 2 or x.dim() == 4:
            output = self.forward(x, proprio, locations=locations)
            return output

        state = x.flatten(0, 1)

        if proprio is not None:
            proprio = proprio.flatten(0, 1)

        if locations is not None:
            locations = locations.flatten(0, 1)

        output = self.forward(state, proprio, locations=locations)

        state = output.encodings
        new_shape = x.shape[:2] + state.shape[1:]
        state = state.reshape(new_shape)

        def reshape_if_not_none(component, x):
            if component is not None:
                new_shape = x.shape[:2] + component.shape[1:]
                return component.reshape(new_shape)
            return None

        obs_component = reshape_if_not_none(output.obs_component, x)
        proprio_component = reshape_if_not_none(output.proprio_component, x)
        location_component = reshape_if_not_none(output.location_component, x)
        raw_locations = reshape_if_not_none(output.raw_locations, x)

        output = BackboneOutput(
            encodings=state,
            obs_component=obs_component,
            proprio_component=proprio_component,
            location_component=location_component,
            raw_locations=raw_locations,
        )

        return output
