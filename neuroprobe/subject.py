from crane import CraneFeature
from crane.data import CraneDataset, Subjects
from crane.preprocess import subset_electrodes
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

        self.dataset = CraneDataset[CraneFeature](
            dataset_dir=cfg.data_dir,
            select=Subjects(subject_id),
            keep_files_open=keep_files_open,
        )

    def load_neural_data(self, trial_id: int, start: float, end: float) -> CraneFeature:
        """Load neural data for the given trial and time window."""
        recording_id = f"sub-{self.subject_id:03}_ses-{trial_id:02}"

        idx = DatasetIndex(recording_id=recording_id, start=start, end=end)
        data = self.dataset[idx]

        data = subset_electrodes(data, subset=self.electrode_subset)
        return data
