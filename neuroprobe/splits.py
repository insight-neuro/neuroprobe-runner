from collections.abc import Iterator, Sequence

import chz
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, Dataset, Subset

from .config import NeuroprobeConfig, NeuroprobeLiteConfig, Task
from .dataset import BrainTreebankDataset
from .subject import BrainTreebankSubject


class _SubsetWithAttr(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int] | np.ndarray):
        super().__init__(dataset, indices)  # type: ignore

    def __getattr__(self, name: str):
        # Delegate attribute access to the underlying dataset
        return getattr(self.dataset, name)


def _val_test_split(dataset: Dataset, val_size: float = 0.5) -> tuple[Dataset, Dataset]:
    """Split a dataset into validation and test sets based."""
    test_size = len(dataset)  # type: ignore
    val_size = int(test_size * val_size)

    val_indices = list(range(val_size))
    test_indices = list(range(val_size, test_size))

    val_ds = _SubsetWithAttr(dataset, val_indices)
    test_ds = _SubsetWithAttr(dataset, test_indices)

    return val_ds, test_ds


def cross_subject_splits(
    cfg: NeuroprobeConfig,
    all_subjects: dict[int, BrainTreebankSubject],
    test_subject_id: int,
    test_trial_id: int,
    task: Task,
    include_all_train_subjects: bool = False,
) -> Iterator[dict[str, Dataset]]:
    """Generate train/test splits for Cross-Subject Task.

    This function creates train/test splits by using one subject and movie as the test set,
    and using all other subjects and movies (except the test movie) as the training set.
    This evaluates generalization across both subjects and movie content (i.e. the same subject but different movies).

    Args:
        all_subjects (dict): Dictionary mapping subject IDs to Subject objects
        test_subject_id (int): ID of the subject to use as test set
        test_trial_id (int): ID of the trial/movie to use as test set

        include_all_other_trials (bool, optional): if True, include all other trials for training (defaults to False). If False, only include subject 2 trial 4 for training (NOTE: for Neuroprobe, there is no choice).

    Returns:
        Iterator[dict[str, Dataset]]: An iterator over dictionaries, each containing:
            - train_dataset (BrainTreebankDataset): Training dataset
            - val_dataset (BrainTreebankDataset): Validation dataset
            - test_dataset (BrainTreebankDataset): Test dataset
    """
    if include_all_train_subjects:
        train_subject_trials = [
            (subject_id, trial_id)
            for subject_id, trial_id in cfg.subject_trials
            if subject_id != test_subject_id
        ]
    else:
        train_subject_trials = [
            (cfg.cross_subject_train_subject_id, cfg.cross_subject_train_trial_id)
        ]
        if test_subject_id == cfg.cross_subject_train_subject_id:
            raise ValueError("Test subject cannot be the same as the training subject.")

    for train_subject_id, train_trial_id in train_subject_trials:
        train_ds = BrainTreebankDataset(
            cfg, all_subjects[test_subject_id], test_trial_id, task
        )

        test_ds = BrainTreebankDataset(
            cfg, all_subjects[train_subject_id], train_trial_id, task
        )

        val_ds, test_ds = _val_test_split(test_ds)

        yield {
            "train_dataset": train_ds,
            "val_dataset": val_ds,
            "test_dataset": test_ds,
        }


def cross_session_splits(
    cfg: NeuroprobeConfig,
    test_subject: BrainTreebankSubject,
    test_trial_id: int,
    task: Task,
    include_all_other_trials: bool = False,
) -> Iterator[dict[str, Dataset]]:
    """Generate train/test splits for Cross-Session Task.

    This function creates train/test splits by using one movie as the test set and all other
    movies from the same subject as the training set (trimmed at max_other_trials movies).
    Unlike Within-Session, this does not perform k-fold cross validation since movies are already naturally separated.

    Args:
        cfg (NeuroprobeConfig): Configuration object containing dataset parameters
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use as test set
        task (Task): The Task name to use for loading the dataset (e.g. "one_second_after_onset")
        include_all_other_trials (bool, optional): if True, include all other trials for training (defaults to False). If False, only include the longest other trial for training.

    Returns:
        Iterator[dict[str, Dataset]]: An iterator over dictionaries, each containing:
            - train_dataset (BrainTreebankDataset): Training dataset
            - val_dataset (BrainTreebankDataset): Validation dataset
            - test_dataset (BrainTreebankDataset): Test dataset
    """

    if len(cfg.longest_trial[test_subject.subject_id]) < 2:
        raise ValueError(
            f"Subject {test_subject.subject_id} does not have enough trials for cross-session splits. At least 2 trials are required, but this subject has {len(cfg.longest_trial[test_subject.subject_id])} trials."
        )

    test_ds = BrainTreebankDataset(cfg, test_subject, test_trial_id, task)

    if include_all_other_trials:
        train_trial_ids = [
            trial_id
            for trial_id in cfg.longest_trial[test_subject.subject_id]
            if trial_id != test_trial_id
        ]

        if cfg.max_samples is not None:
            cfg = chz.replace(cfg, max_samples=cfg.max_samples // len(train_trial_ids))

        train_datasets = [
            BrainTreebankDataset(cfg, test_subject, train_trial_id, task)
            for train_trial_id in train_trial_ids
        ]
        train_ds = ConcatDataset(train_datasets)
    else:
        if not isinstance(cfg, NeuroprobeLiteConfig):
            train_trial_id = cfg.longest_trial[test_subject.subject_id][0]
            if train_trial_id == test_trial_id:
                train_trial_id = cfg.longest_trial[test_subject.subject_id][1]
                # If the longest trial is the test trial, use the second longest trial for training
        else:
            train_trial_id = [
                trial_id
                for subject_id, trial_id in cfg.subject_trials
                if subject_id == test_subject.subject_id and trial_id != test_trial_id
            ][0]
            # Get the first other trial for the training set (there should only be one)

        train_ds = BrainTreebankDataset(cfg, test_subject, train_trial_id, task)

    val_ds, test_ds = _val_test_split(test_ds)

    yield {
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


def within_session_splits(
    cfg: NeuroprobeConfig, test_subject, test_trial_id, task: Task
) -> Iterator[dict[str, Dataset]]:
    """Generate train/test splits for Within Session Task.

    This function performs k-fold cross validation on data from a single subject and movie.


    Args:
        test_subject (Subject): Subject object containing brain recording data
        test_trial_id (int): ID of the trial/movie to use
        task (Task): The Task to use for loading the dataset (e.g. "one_second_after_onset")

    Returns:
        list: A list of dictionaries, each containing:
            - train_dataset (Dataset): Training dataset
            - val_dataset (Dataset): Validation dataset
            - test_dataset (Dataset): Test dataset
    """

    dataset = BrainTreebankDataset(cfg, test_subject, test_trial_id, task)

    k_folds = cfg.num_cv_folds
    kf = KFold(n_splits=k_folds, shuffle=False)
    # shuffle=False is important to avoid correlated train/test splits!

    for train_idx, test_idx in kf.split(dataset):  # type: ignore
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        train_ds = _SubsetWithAttr(dataset, train_idx)
        test_ds = _SubsetWithAttr(dataset, test_idx)
        val_ds, test_ds = _val_test_split(test_ds)

        yield {
            "train_dataset": train_ds,
            "val_dataset": val_ds,
            "test_dataset": test_ds,
        }
