"""
Wearable Data Analysis Package

A comprehensive package for analyzing wearable sensor data and classifying physical activities.
"""

__version__ = "1.0.0"
__author__ = "Wearable Data Analysis Team"

from .data_loader import DataLoader
from .feature_engineer import FeatureEngineer
from .model import WearableModel
from .evaluator import ModelEvaluator
from .visualizer import ResultVisualizer
from .load_uci_har import UCIHARLoader, load_uci_har_dataset, load_uci_har_for_pipeline

__all__ = [
    "DataLoader",
    "FeatureEngineer",
    "WearableModel",
    "ModelEvaluator",
    "ResultVisualizer",
    "UCIHARLoader",
    "load_uci_har_dataset",
    "load_uci_har_for_pipeline"
]