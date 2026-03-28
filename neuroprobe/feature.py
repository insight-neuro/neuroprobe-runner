from crane.core.featurizer import BrainFeature
from crane.data.structures import ChannelDict
from jaxtyping import Float
from torch import Tensor


class NeuroprobeFeature(BrainFeature):
    """Feature class for Neuroprobe"""

    ieeg: Float[Tensor, "C T"]
    """Intracranial EEG data tensor of shape [num_channels, num_timepoints]"""

    channels: ChannelDict
    """Channel metadata containing ids, locations, and other info for each electrode"""

    sampling_rate: float
    """Sampling rate of the neural data in Hz (2800 Hz for Neuroprobe)"""

    label: int
    """Supervised label for the sample (e.g., 0 or 1 for binary classification)"""
