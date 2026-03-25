"""
Neuroprobe: A benchmark for evaluating intracranial brain responses to naturalistic stimuli.

This package provides tools for analyzing neural data from the BrainTreebank dataset,
including dataset loading, preprocessing, and evaluation utilities.
"""

__version__ = "0.2.0"
__author__ = "Andrii Zahorodnii, Christopher Wang, Bennett Stankovits, Charikleia Moraitaki, Geeling Chau, Andrei Barbu, Boris Katz, Ila R Fiete"
__email__ = "zaho@csail.mit.edu"

from .config import (
    NeuroprobeConfig,
    NeuroprobeLiteConfig,
    NeuroprobeNanoConfig,
    NeurprobeFullConfig,
)
from .dataset import BrainTreebankDataset
from .runner import NeuroprobeRunner
from .splits import (
    cross_session_splits,
    cross_subject_splits,
    within_session_splits,
)
from .subject import BrainTreebankSubject

__all__ = [
    "NeuroprobeLiteConfig",
    "NeuroprobeNanoConfig",
    "NeuroprobeConfig",
    "NeurprobeFullConfig",
    "cross_session_splits",
    "cross_subject_splits",
    "within_session_splits",
    "BrainTreebankSubject",
    "NeuroprobeRunner",
    "BrainTreebankDataset",
]
