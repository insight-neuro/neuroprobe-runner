import os
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

import chz
import torch
from chz.validators import ge, lt

from .load import electrodes

type RegressionTask = Literal[
    "frame_brightness",
    "global_flow",
    "local_flow",
    "face_num",
    "volume",
    "pitch",
    "delta_volume",
    "speech",
    "onset",
    "gpt2_surprisal",
    "word_length",
    "word_gap",
    "word_index",
]

type ClassificationTask = Literal[
    "word_head_pos",
    "word_part_speech",
]

type Task = RegressionTask | ClassificationTask

type split = Literal["within_session", "cross_subject", "cross_session"]


@chz.chz(typecheck=True)
class NeuroprobeConfig:
    # ======= Paths =======

    data_dir: Path = chz.field(
        default_factory=lambda: Path(os.environ["ROOT_DIR_BRAINTREEBANK"]),
        doc="Root directory of the Braintreebank dataset.",
    )

    results_dir: Path = Path("results")
    """Directory where evaluation outputs are stored.
    Should be shared across runs for leaderboard comparability."""

    # ======= Run Settings =======

    tasks: list[Task] = chz.field(default_factory=lambda: get_args(Task))
    """Tasks to evaluate. Defaults to all available tasks."""

    eval_splits: list[split] = chz.field(default_factory=lambda: get_args(split))
    """Evaluation splits to run:\n- within_session\n- cross_session\n- cross_subject\n\nDefaults to all splits."""

    # ======= Evaluation Settings =======

    tensor_dtype: torch.dtype | str = torch.float32
    """Torch dtype for returned tensors."""

    use_binary_targets: bool = True
    """If True → binary classification. If False → multi-class tasks (see paper for details)."""

    num_cv_folds: int = 2
    """Number of folds for cross-validation for within-session evaluation."""

    max_samples: int | None = 3500
    """Max number of samples to use in a dataset. If None, use all samples"""

    subject_trials: list[tuple[int, int]] = [  # structured as (subject_id, trial_id)
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 4),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
        (7, 0),
        (7, 1),
        (10, 0),
        (10, 1),
    ]
    """Subjects and trials to use for evaluation, structured as a list of (subject_id, trial_id) tuples."""

    random_seed: int = 42
    """Global RNG seed for reproducibility."""

    # ======= Cross-subject protocol =======

    cross_subject_train_subject_id: int = 2
    """Standard training subject for cross-subject evaluation."""

    cross_subject_train_trial_id: int = 4
    """Standard training subject for cross-subject evaluation."""

    # =============================================
    # ⚠️ FIXED PROTOCOL — DO NOT MODIFY
    # DO NOT MODIFY for leaderboard comparability
    # =============================================

    sampling_rate: int = 2048
    """Sampling rate in Hertz (do not change this)"""

    word_onset_window_start: float = 0
    """Start time relative to word onset (seconds). Overriden to 0 for the 1-second evaluation on the leaderboard."""

    word_onset_window_end: float = 1
    """End time relative to word onset (seconds). Overriden to 1 for the 1-second evaluation on the leaderboard."""

    nonverbal_gap: float = 2
    """Gap between speech and non-verbal segments (seconds)."""

    nonverbal_overlap_ratio: float = chz.field(
        default=0.5,
        validator=[ge(0), lt(1)],
    )
    """Overlap ratio between consecutive non-verbal windows"""

    # ====== Structural information about the dataset ======

    electrodes: dict[str, list[str]] = chz.field(
        default_factory=lambda: electrodes("lite")
    )
    """For each subject, the list of electrodes to use for evaluation. Structured as a dict: subject_id -> list of electrode names."""

    longest_trial: MappingProxyType[int, list[int]] = MappingProxyType(
        {
            # structured as subject_id: [longest_trial_id, second_longest_trial_id]
            1: [0, 1],
            2: [4, 6],
            3: [2, 1],
            4: [2, 1],
            5: [0],
            6: [0, 2],
            7: [1, 0],
            8: [0],
            9: [0],
            10: [1, 0],
        }
    )
    """Per-subject longest trials mapping: subject_id -> [longest_trial_id, second_longest_trial_id]"""


@chz.chz
class NeuroprobeLiteConfig(NeuroprobeConfig): ...  # Equivalent to base config.


@chz.chz
class NeuroprobeNanoConfig(NeuroprobeConfig):
    max_samples = 1000
    num_cv_folds = 2
    subject_trials = [
        (1, 1),
        (2, 4),
        (3, 1),
        (4, 0),
        (7, 1),
        (10, 1),
    ]
    electrodes: dict[str, list[str]] = chz.field(
        default_factory=lambda: electrodes("nano")
    )


class NeurprobeFullConfig(NeuroprobeConfig):
    max_samples = None
    subject_trials = [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (6, 0),
        (6, 1),
        (6, 4),
        (7, 0),
        (7, 1),
        (8, 0),
        (9, 0),
        (10, 0),
        (10, 1),
    ]
