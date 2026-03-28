import torch
from crane.data import CraneDataset, Subjects
from crane.data.structures import ChannelDict
from crane.preprocess import subset_electrodes
from jaxtyping import Float
from torch import Tensor
from torch_brain.dataset import DatasetIndex

from .config import NeuroprobeConfig


class BrainTreebankSubject:
    """Thin wrapper around the CraneDataset for loading neural data for a given subject.

    Args:
        cfg: NeuroprobeConfig object containing dataset settings.
        subject_id: ID of the subject to load (e.g., 1, 2, 3, etc.).
        keep_files_open: Whether to keep HDF5 files open for faster access (default: True).
    """

    def __init__(
        self,
        cfg: NeuroprobeConfig,
        subject_id: int,
        keep_files_open: bool = True,
    ):
        self.subject_id = subject_id
        self.electrode_subset = cfg.electrodes[f"btbank{subject_id}"]

        self.dataset = CraneDataset(
            dataset_dir=cfg.data_dir,
            select=Subjects(subject_id),
            keep_files_open=keep_files_open,
        )

    def load_neural_data(
        self, trial_id: int, start: float, end: float
    ) -> tuple[Float[Tensor, "n_channels n_timepoints"], ChannelDict]:
        """Load neural data for the given trial and time window."""
        recording_id = f"sub-{self.subject_id:03}_ses-{trial_id:02}"

        idx = DatasetIndex(recording_id=recording_id, start=start, end=end)
        data = self.dataset[idx]

        ieeg = torch.from_numpy(data["data"].data.T)  # [n_electrodes, n_timebins]
        channels = ChannelDict(**data["channels"].materialize().__dict__)
        ieeg, channels = subset_electrodes(ieeg, channels, subset=self.electrode_subset)

        return ieeg, channels
