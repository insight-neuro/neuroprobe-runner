import json
from functools import cache
from pathlib import Path
from typing import Literal

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data"


@cache
def electrodes(
    subset: Literal["nano", "lite", "missing_coordinate"],
) -> dict[str, list[str]]:
    with open(DATA_PATH / f"{subset}_electrodes.json") as f:
        electrodes = json.load(f)
    return electrodes


@cache
def trial_movies() -> dict[str, str]:
    with open(DATA_PATH / "trial_movies.json") as f:
        mapping = json.load(f)
    return mapping


def get_movie_name(subject_id: int, trial_id: int) -> str:
    return trial_movies()[f"btbank{subject_id}_{trial_id}"]


def time_alignment_features(
    subject_id: int, trial_id: int, features: Literal["words", "nonverbal"]
) -> pd.DataFrame:
    return pd.read_csv(
        DATA_PATH
        / f"time_alignment_features/subject{subject_id}_trial{trial_id}_{features}_df.csv"
    )


def pitch_volume_features(movie_name: str) -> dict:
    path = DATA_PATH / f"pitch_volume_features/{movie_name}_pitch_volume_features.json"
    with open(path) as f:
        features = json.load(f)
    return features
