import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from neuroprobe.dataset import BrainTreebankDataset

from .config import NeuroprobeConfig
from .splits import cross_session_splits, cross_subject_splits, within_session_splits
from .subject import BrainTreebankSubject

logger = logging.getLogger(__name__)


class NeuroprobeRunner(ABC):
    """Interface for running the standard Neuroprobe evaluation for leaderboard submission.

    Users should fill in the following class variables to provide metadata about their model:

    - model_name: The name of the model being evaluated.
    - description: A short description of the model/method.
    - author: The name of the author.
    - organization: The author's affiliation.
    - organization_url: The homepage of the organization.

    Additionally, the users must implement the following class method:

    - evaluate_fold(cls, train_ds, val_ds, test_ds) -> dict[str, Any]:
        This method should define the evaluation procedure for a single fold of the data and return a dictionary of evaluation results.

    To run, the easiest way is using chz's nested entry point, as it provides a CLI:

    ```py
    class Runner(NeuroprobeRunner): ...

    chz.nested_entrypoint(Runner.run)
    ```

    Or otherwise, with a configuration object:

    ```py
    cfg = NeuroprobeConfig(...)
    Runner.run(cfg)
    ```
    """

    model_name: ClassVar[str]
    """Model evaluated."""
    description: ClassVar[str]
    """Short description of the model/method."""
    author: ClassVar[str]
    """Author's name."""
    organization: ClassVar[str]
    """Author's affiliation."""
    organization_url: ClassVar[str]
    """Organization's homepage."""
    organization_logo: ClassVar[str | None] = None
    """Link to organization's logo (optional)."""

    @classmethod
    def run(cls, cfg: NeuroprobeConfig): 
        """Run the Neuroprobe evaluation according to the provided configuration.

        Args:
            cfg (NeuroprobeConfig): Configuration for the evaluation, including dataset paths, evaluation splits, tasks, and other parameters.
        """

        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)

        today = datetime.today().strftime("%Y%m%d")
        root_dir = Path(cfg.results_dir) / f"{cls.model_name}_{cls.author}_{today}"

        data_idx_from = int(cfg.word_onset_window_start * cfg.sampling_rate)
        data_idx_to = int(cfg.word_onset_window_end * cfg.sampling_rate)

        logger.info(
            "Starting evaluation | model=%s | splits=%s | tasks=%s | output=%s",
            cls.model_name,
            cfg.eval_splits,
            cfg.tasks,
            root_dir,
        )

        for split in cfg.eval_splits:
            split_dir = root_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Processing split='%s' → %s", split, split_dir)

            for task in cfg.tasks:
                file_path = split_dir / f"population_{task}.json"

                skip_trials: set[str] = set()
                if file_path.exists():
                    with open(file_path) as f:
                        results = json.load(f)
                    skip_trials.update(results.get("evaluation_results", {}).keys())

                    logger.info(
                        "Resuming existing results | file=%s | skipping=%d trials",
                        file_path,
                        len(skip_trials),
                    )

                else:
                    logger.info("Initializing new results file → %s", file_path)
                    results = {
                        "model_name": cls.model_name,
                        "author": cls.author,
                        "description": cls.description,
                        "organization": cls.organization,
                        "organization_url": cls.organization_url,
                        "timestamp": time.time(),
                        "evaluation_results": {},
                    }

                for subject_id, trial_id in cfg.subject_trials:
                    trial_session_tag = f"btbank{subject_id}_{trial_id}"
                    if trial_session_tag in skip_trials:
                        logger.debug(
                            "Skipping already computed trial=%s", trial_session_tag
                        )
                        continue

                    logger.info(
                        "Running trial | subject=%s | trial=%s | split=%s | task=%s",
                        subject_id,
                        trial_id,
                        split,
                        task,
                    )

                    subject = BrainTreebankSubject(cfg, subject_id)

                    train_subject = subject
                    match split:
                        case "within_session":
                            folds = within_session_splits(
                                cfg, subject, trial_id, task=task
                            )
                        case "cross_session":
                            folds = cross_session_splits(
                                cfg, subject, trial_id, task=task
                            )
                        case "cross_subject":
                            train_subject_id = cfg.cross_subject_train_subject_id
                            train_subject = BrainTreebankSubject(cfg, train_subject_id)
                            if subject_id == train_subject_id:
                                logger.warning(
                                    "Skipping cross-subject | subject=%s equals train_subject=%s",
                                    subject_id,
                                    train_subject_id,
                                )
                                continue

                            all_subjects = {
                                subject_id: subject,
                                train_subject_id: train_subject,
                            }
                            folds = cross_subject_splits(
                                cfg,
                                all_subjects,
                                subject_id,
                                trial_id,
                                task=task,
                            )

                    trial_results = {
                        "time_bin_start": data_idx_from,
                        "time_bin_end": data_idx_to,
                        "folds": [],
                    }

                    for fold in folds:
                        fold_results = cls.evaluate_fold(
                            fold["train_dataset"],
                            fold["val_dataset"],
                            fold["test_dataset"],
                        )
                        trial_results["folds"].append(fold_results)

                    results["evaluation_results"].setdefault(trial_session_tag, {})
                    results["evaluation_results"][trial_session_tag]["population"] = {
                        "one_second_after_onset": trial_results
                    }

                    logger.info("Completed trial=%s", trial_session_tag)

                with open(file_path, "w") as f:
                    json.dump(results, f, indent=4)

                logger.info("Saved results → %s", file_path)

    @classmethod
    @abstractmethod
    def evaluate_fold(
        cls,
        train_ds: BrainTreebankDataset,
        val_ds: BrainTreebankDataset,
        test_ds: BrainTreebankDataset
    ) -> dict[str, Any]:
        """This method should be implemented by subclasses to define the
        specific evaluation procedure for a single fold of the data.

        Args:
            train_ds (BrainTreebankDataset): The training dataset.
            val_ds (BrainTreebankDataset): The validation dataset.
            test_ds (BrainTreebankDataset): The test dataset.

        Returns:
            A dictionary containing the evaluation results, e.g.
            `{"train_accuracy": 0.9, "test_accuracy": 0.85, "train_auroc": 0.95, "test_auroc": 0.9}`
        """
        pass
