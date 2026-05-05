from pathlib import Path

import numpy as np
import pandas as pd
import torch
from crane import CraneFeature

from .annotations import get_movie_name, pitch_volume_features, time_alignment_features
from .config import NeuroprobeConfig, Task
from .subject import BrainTreebankSubject

# Defining the names of evaluations and preparing them for downstream processing
single_float_variables_name_remapping = {
    "pitch": "enhanced_pitch",
    "volume": "rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length",
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos",
}
new_pitch_variables = [
    "enhanced_pitch",
    "enhanced_volume",
    "delta_enhanced_pitch",
    "delta_enhanced_volume",
    "raw_pitch",
    "raw_volume",
    "delta_raw_pitch",
    "delta_raw_volume",
]
single_float_variables = (
    list(single_float_variables_name_remapping.values())
    + list(single_float_variables_name_remapping.keys())
    + new_pitch_variables
)
classification_variables = list(
    classification_variables_name_remapping.values()
) + list(classification_variables_name_remapping.keys())
all_tasks = (
    single_float_variables
    + ["onset", "speech"]
    + ["face_num", "word_gap", "word_index"]
    + classification_variables
)


class NeuroprobeFeature(CraneFeature):
    """CraneFeature subclass for the Neuroprobe dataset, with additional attributes for task labels."""

    label: int
    """Task label for the sample, e.g., 0 or 1 for binary classification."""


class BrainTreebankDataset(torch.utils.data.Dataset):
    """Dataset for a subject-trial pair."""

    def __init__(
        self,
        cfg: NeuroprobeConfig,
        subject: BrainTreebankSubject,
        trial_id: int,
        task: Task,
    ):
        """
        Args:
            cfg (NeuroprobeConfig): Configuration object containing dataset settings.
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            task (Task): the task to evaluate on (e.g. "pitch", "word_head_pos", etc.)
        """

        # Set up a local random state with the provided seed
        self.rng = np.random.RandomState(cfg.random_seed)

        if task not in all_tasks:
            raise ValueError(f"Task must be one of {all_tasks}, not {task}")
        
        self.cfg = cfg
        self.subject = subject
        self.trial_id = trial_id
        self.task = task

        self._build_label_indices()

    def _build_label_indices(self):
        task_remapped = self.task
        if self.task in single_float_variables_name_remapping:
            task_remapped = single_float_variables_name_remapping[self.task]
        if self.task in classification_variables_name_remapping:
            task_remapped = classification_variables_name_remapping[self.task]

        self.all_words_df = time_alignment_features(
            self.subject.subject_id, self.trial_id, "words"
        )
        self.nonverbal_df = time_alignment_features(
            self.subject.subject_id, self.trial_id, "nonverbal"
        )

        movie_name = get_movie_name(self.subject.subject_id, self.trial_id)

        # Add the original features from braintreebank to the all_words_df
        transcript_file_path = (
            Path(self.cfg.data_dir) / f"transcripts/{movie_name}/features.csv"
        )
        original_features_df = pd.read_csv(transcript_file_path).set_index("Unnamed: 0")

        # Add new columns from words_df using original_index mapping
        new_columns = [
            col
            for col in original_features_df.columns
            if col not in self.all_words_df.columns
        ]
        for col in new_columns:
            self.all_words_df[col] = self.all_words_df["original_index"].map(
                original_features_df[col]
            )

        if self.task in single_float_variables:
            # Grab the new pitch volume features if they exist
            if task_remapped in new_pitch_variables:
                raw_pitch_volume_features = pitch_volume_features(movie_name)

                TARGET_DP_FOR_KEYS = 5  # Standard number of decimal places
                normalized_pvf = {}
                for k_str, v_val in raw_pitch_volume_features.items():
                    k_float = float(k_str)
                    normalized_key = f"{k_float:.{TARGET_DP_FOR_KEYS}f}"
                    normalized_pvf[normalized_key] = v_val
                features = normalized_pvf

                start_times = self.all_words_df["start"].to_list()
                all_labels = []
                for start_time_val in start_times:
                    lookup_key = f"{start_time_val:.{TARGET_DP_FOR_KEYS}f}"
                    label = features[lookup_key][task_remapped]
                    all_labels.append(label)
                all_labels = np.array(all_labels)
            else:
                all_labels = self.all_words_df[task_remapped].to_numpy()

            # Get indices for words in top and bottom quartiles
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            if self.cfg.use_binary_targets:
                self.label_indices = {
                    1: np.where(label_percentiles > 0.75)[0],
                    0: np.where(label_percentiles < 0.25)[0],
                }
            else:
                self.label_indices = {
                    2: np.where(label_percentiles >= 0.75)[0],
                    1: np.where(
                        (label_percentiles < 0.625) & (label_percentiles >= 0.375)
                    )[0],
                    0: np.where(label_percentiles < 0.25)[0],
                }
        elif self.task in ["onset", "speech"]:
            self.label_indices = {
                1: np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0]
                if self.task == "onset"
                else np.arange(len(self.all_words_df)),  # positive indices
                0: np.arange(len(self.nonverbal_df)),  # negative indices
            }
        elif self.task == "face_num":
            face_nums = self.all_words_df["face_num"].to_numpy().astype(int)
            if self.cfg.use_binary_targets:
                self.label_indices = {
                    1: np.where(face_nums > 0)[0],
                    0: np.where(face_nums == 0)[0],
                }
            else:
                self.label_indices = {
                    2: np.where(face_nums > 1)[0],
                    1: np.where(face_nums == 1)[0],
                    0: np.where(face_nums == 0)[0],
                }
        elif self.task == "word_index":
            word_indices = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            if self.cfg.use_binary_targets:
                self.label_indices = {
                    1: np.where(word_indices == 0)[0],
                    0: np.where(word_indices == 1)[0],
                }
            else:
                self.label_indices = {
                    2: np.where(word_indices >= 2)[0],
                    1: np.where(word_indices == 1)[0],
                    0: np.where(word_indices == 0)[0],
                }
        elif self.task == "word_head_pos":
            head_pos = self.all_words_df[task_remapped].to_numpy().astype(int)
            self.label_indices = {
                1: np.where(head_pos == 0)[0],
                0: np.where(head_pos == 1)[0],
            }
        elif self.task == "word_part_speech":
            pos = self.all_words_df[task_remapped].to_numpy()
            if self.cfg.use_binary_targets:
                self.label_indices = {
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0],
                }
            else:
                self.label_indices = {
                    5: np.where(pos == "ADV")[0],
                    4: np.where(pos == "ADJ")[0],
                    3: np.where(pos == "DET")[0],
                    2: np.where(pos == "PRON")[0],
                    1: np.where(pos == "VERB")[0],
                    0: np.where(pos == "NOUN")[0],
                }
        elif self.task == "word_gap":
            word_gap_distribution = []
            for i in range(1, len(self.all_words_df)):
                if (
                    self.all_words_df.iloc[i]["sentence"]
                    != self.all_words_df.iloc[i - 1]["sentence"]
                ):
                    continue
                gap = (
                    self.all_words_df.iloc[i]["start"]
                    - self.all_words_df.iloc[i - 1]["end"]
                )
                word_gap_distribution.append(gap)
            word_gap_distribution = np.array(word_gap_distribution)

            positive_indices = []
            negative_indices = []
            middle_indices = []
            for i in range(1, len(self.all_words_df)):
                if (
                    self.all_words_df.iloc[i]["sentence"]
                    != self.all_words_df.iloc[i - 1]["sentence"]
                ):
                    continue
                gap = (
                    self.all_words_df.iloc[i]["start"]
                    - self.all_words_df.iloc[i - 1]["end"]
                )
                gap_percentile = np.mean(word_gap_distribution < gap)
                if gap_percentile >= 0.75:
                    positive_indices.append(i)
                elif (gap_percentile >= 0.375) and (gap_percentile < 0.625):
                    middle_indices.append(i)
                elif gap_percentile < 0.25:
                    negative_indices.append(i)
            if self.cfg.use_binary_targets:
                self.label_indices = {1: positive_indices, 0: negative_indices}
            else:
                self.label_indices = {
                    2: positive_indices,
                    1: middle_indices,
                    0: negative_indices,
                }
        else:
            raise ValueError(f"Invalid task: {self.task}")

        self.n_classes = len(self.label_indices)
        n_samples_each = min(
            len(self.label_indices[label]) for label in self.label_indices
        )

        if self.cfg.max_samples is not None:
            n_samples_each = min(n_samples_each, self.cfg.max_samples // self.n_classes)

        for label in list(self.label_indices.keys()):
            self.label_indices[label] = np.sort(  # type: ignore
                self.rng.choice(
                    self.label_indices[label], size=n_samples_each, replace=False
                )
            )
            if (
                self.cfg.max_samples is not None
            ):  # if max_samples is set, we need to truncate the indices to the max_samples
                self.label_indices[label] = self.label_indices[label][  # type: ignore
                    : self.cfg.max_samples // self.n_classes
                ]

        self.n_samples = sum(
            [len(self.label_indices[label]) for label in self.label_indices]
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> NeuroprobeFeature:
        if idx >= self.n_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.n_samples}"
            )

        # Even indices -> positive samples, Odd indices -> negative samples
        label = (idx + 1) % self.n_classes
        word_index = self.label_indices[label][idx // self.n_classes]

        if self.task in ["onset", "speech"] and label == 0:
            # for onset and speech, we need to get the nonverbal data
            row = self.nonverbal_df.iloc[word_index]
        else:
            row = self.all_words_df.iloc[word_index]

        est_idx = row["est_idx"] // self.cfg.sampling_rate
        start_time = est_idx - self.cfg.word_onset_window_start
        end_time = est_idx + self.cfg.word_onset_window_end

        feat = self.subject.load_neural_data(
            self.trial_id, start=start_time, end=end_time
        )
        feat["label"] = label
        return feat.to(self.cfg.tensor_dtype)  # type: ignore[return-value]

    @property
    def signals(self) -> np.ndarray:
        """Neural signals for all samples in the dataset, as a numpy array of shape (num_samples, num_channels, num_timepoints)."""
        return np.array([feature.signals.float().numpy() for feature in self])

    @property
    def labels(self) -> np.ndarray:
        """Labels for all samples in the dataset, as a numpy array of shape (num_samples,)."""
        return np.array([feature.label for feature in self])
