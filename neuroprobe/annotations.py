import json
from functools import cache
from importlib.resources import files
from typing import Literal

import pandas as pd

DATA_PATH = files("neuroprobe") / "annotations"


@cache
def electrodes(
    subset: Literal["nano", "lite", "missing_coordinate"],
) -> dict[str, list[str]]:
    with (DATA_PATH / f"{subset}_electrodes.json").open("r") as f:
        electrodes = json.load(f)
    return electrodes


@cache
def trial_movies() -> dict[str, str]:
    with (DATA_PATH / "trial_movies.json").open("r") as f:
        mapping = json.load(f)
    return mapping


def get_movie_name(subject_id: int, trial_id: int) -> str:
    return trial_movies()[f"btbank{subject_id}_{trial_id}"]


def time_alignment_features(
    subject_id: int, trial_id: int, features: Literal["words", "nonverbal"]
) -> pd.DataFrame:
    with (
        DATA_PATH
        / "time_alignment_features"
        / f"subject{subject_id}_trial{trial_id}_{features}_df.csv"
    ).open("r") as f:
        df = pd.read_csv(f)
    return df


def pitch_volume_features(movie_name: str) -> dict:
    path = (
        DATA_PATH / "pitch_volume_features" / f"{movie_name}_pitch_volume_features.json"
    )
    with (path).open("r") as f:
        features = json.load(f)
    return features
