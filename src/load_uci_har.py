"""
UCI HAR Dataset Loader

Enhanced loader for the UCI Human Activity Recognition dataset that integrates
seamlessly with the wearable data analysis pipeline.

Dataset: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict, Any
import requests
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UCIHARLoader:
    """
    Enhanced UCI HAR dataset loader with automatic download and preprocessing.

    The UCI HAR dataset contains sensor data from smartphones worn by 30 volunteers
    performing six activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, 
    SITTING, STANDING, LAYING.
    """

    def __init__(self, data_dir: str = "data/raw/UCI_HAR"):
        """
        Initialize UCI HAR loader.

        Args:
            data_dir: Directory to store/load UCI HAR dataset
        """
        self.data_dir = Path(data_dir)
        self.dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        self.extracted_dir = self.data_dir / "UCI HAR Dataset"

        # UCI HAR activity mapping
        self.activity_mapping = {
            1: "WALKING",
            2: "WALKING_UPSTAIRS",
            3: "WALKING_DOWNSTAIRS",
            4: "SITTING",
            5: "STANDING",
            6: "LAYING"
        }

        # Sensor information
        self.sensor_info = {
            'sampling_rate': 50,  # Hz
            'window_size': 2.56,  # seconds
            'overlap': 0.5,  # 50% overlap
            'n_samples_per_window': 128
        }

    def download_dataset(self, force_download: bool = False) -> bool:
        """
        Download UCI HAR dataset if not already present.

        Args:
            force_download: Force re-download even if dataset exists

        Returns:
            True if dataset is available, False otherwise
        """
        if self.extracted_dir.exists() and not force_download:
            logger.info(f"UCI HAR dataset already exists at {self.extracted_dir}")
            return True

        logger.info("Downloading UCI HAR dataset...")

        try:
            # Create directory
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Download zip file
            zip_path = self.data_dir / "UCI_HAR_Dataset.zip"

            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading: {progress:.1f}%", end="", flush=True)

            print()  # New line after progress
            logger.info(f"Downloaded {downloaded} bytes")

            # Extract zip file
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Clean up zip file
            zip_path.unlink()

            logger.info(f"UCI HAR dataset extracted to {self.extracted_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to download UCI HAR dataset: {str(e)}")
            return False

    def load_uci_har_dataset(self, path: Optional[str] = None,
                             include_raw_signals: bool = False) -> pd.DataFrame:
        """
        Load UCI HAR dataset with enhanced preprocessing.

        Args:
            path: Custom path to UCI HAR dataset (if None, uses self.extracted_dir)
            include_raw_signals: Whether to include raw sensor signals

        Returns:
            DataFrame with processed UCI HAR data
        """
        if path is None:
            path = str(self.extracted_dir)

        path = Path(path)

        if not path.exists():
            logger.warning(f"Dataset not found at {path}. Attempting to download...")
            if not self.download_dataset():
                raise FileNotFoundError(f"Could not find or download UCI HAR dataset at {path}")
            path = self.extracted_dir

        logger.info(f"Loading UCI HAR dataset from {path}")

        try:
            # Load features
            feature_names = pd.read_csv(
                path / "features.txt",
                sep=r"\s+",
                header=None,
                names=["index", "feature"]
            )
            feature_names = feature_names["feature"].tolist()

            # Load activity labels
            activity_labels = pd.read_csv(
                path / "activity_labels.txt",
                sep=r"\s+",
                header=None,
                names=["id", "label"]
            )
            activity_labels = dict(zip(activity_labels.id, activity_labels.label))

            # Load train data
            X_train = pd.read_csv(
                path / "train" / "X_train.txt",
                sep=r"\s+",
                header=None,
                names=feature_names
            )
            y_train = pd.read_csv(
                path / "train" / "y_train.txt",
                sep=r"\s+",
                header=None,
                names=["Activity"]
            )
            subject_train = pd.read_csv(
                path / "train" / "subject_train.txt",
                sep=r"\s+",
                header=None,
                names=["Subject"]
            )

            # Combine train
            train = pd.concat([subject_train, y_train, X_train], axis=1)
            train['dataset_split'] = 'train'

            # Load test data
            X_test = pd.read_csv(
                path / "test" / "X_test.txt",
                sep=r"\s+",
                header=None,
                names=feature_names
            )
            y_test = pd.read_csv(
                path / "test" / "y_test.txt",
                sep=r"\s+",
                header=None,
                names=["Activity"]
            )
            subject_test = pd.read_csv(
                path / "test" / "subject_test.txt",
                sep=r"\s+",
                header=None,
                names=["Subject"]
            )

            # Combine test
            test = pd.concat([subject_test, y_test, X_test], axis=1)
            test['dataset_split'] = 'test'

            # Merge train + test
            df = pd.concat([train, test], axis=0).reset_index(drop=True)

            # Map activity IDs to labels
            df["Activity"] = df["Activity"].map(activity_labels)

            # Add metadata
            df['sampling_rate'] = self.sensor_info['sampling_rate']
            df['window_size_seconds'] = self.sensor_info['window_size']

            # Add raw signals if requested
            if include_raw_signals:
                raw_signals = self._load_raw_signals(path)
                if raw_signals is not None:
                    df = pd.merge(df, raw_signals, left_index=True, right_index=True, how='left')

            logger.info(f"Loaded UCI HAR dataset: {df.shape}")
            logger.info(f"Activities: {df['Activity'].value_counts().to_dict()}")
            logger.info(f"Subjects: {df['Subject'].nunique()} unique subjects")

            return df

        except Exception as e:
            logger.error(f"Error loading UCI HAR dataset: {str(e)}")
            raise

    def _load_raw_signals(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Load raw sensor signals from UCI HAR dataset.

        Args:
            path: Path to UCI HAR dataset

        Returns:
            DataFrame with raw signals or None if not available
        """
        try:
            logger.info("Loading raw sensor signals...")

            raw_signals = {}
            signal_files = [
                ('body_acc_x', 'train/Inertial Signals/body_acc_x_train.txt'),
                ('body_acc_y', 'train/Inertial Signals/body_acc_y_train.txt'),
                ('body_acc_z', 'train/Inertial Signals/body_acc_z_train.txt'),
                ('total_acc_x', 'train/Inertial Signals/total_acc_x_train.txt'),
                ('total_acc_y', 'train/Inertial Signals/total_acc_y_train.txt'),
                ('total_acc_z', 'train/Inertial Signals/total_acc_z_train.txt'),
                ('body_gyro_x', 'train/Inertial Signals/body_gyro_x_train.txt'),
                ('body_gyro_y', 'train/Inertial Signals/body_gyro_y_train.txt'),
                ('body_gyro_z', 'train/Inertial Signals/body_gyro_z_train.txt'),
            ]

            # Load training signals
            for signal_name, file_path in signal_files:
                signal_path = path / file_path
                if signal_path.exists():
                    signal_data = pd.read_csv(signal_path, sep=r"\s+", header=None)
                    # Calculate mean, std, max for each window
                    raw_signals[f'{signal_name}_mean'] = signal_data.mean(axis=1)
                    raw_signals[f'{signal_name}_std'] = signal_data.std(axis=1)
                    raw_signals[f'{signal_name}_max'] = signal_data.max(axis=1)

            # Load test signals
            test_signal_files = [f.replace('train', 'test') for _, f in signal_files]

            for i, (signal_name, _) in enumerate(signal_files):
                test_file_path = test_signal_files[i]
                signal_path = path / test_file_path
                if signal_path.exists():
                    signal_data = pd.read_csv(signal_path, sep=r"\s+", header=None)
                    # Append test data
                    train_len = len(raw_signals[f'{signal_name}_mean'])
                    raw_signals[f'{signal_name}_mean'] = pd.concat([
                        raw_signals[f'{signal_name}_mean'],
                        signal_data.mean(axis=1)
                    ]).reset_index(drop=True)
                    raw_signals[f'{signal_name}_std'] = pd.concat([
                        raw_signals[f'{signal_name}_std'],
                        signal_data.std(axis=1)
                    ]).reset_index(drop=True)
                    raw_signals[f'{signal_name}_max'] = pd.concat([
                        raw_signals[f'{signal_name}_max'],
                        signal_data.max(axis=1)
                    ]).reset_index(drop=True)

            if raw_signals:
                signals_df = pd.DataFrame(raw_signals)
                logger.info(f"Loaded raw signals: {signals_df.shape}")
                return signals_df
            else:
                logger.warning("No raw signals found")
                return None

        except Exception as e:
            logger.warning(f"Could not load raw signals: {str(e)}")
            return None

    def convert_to_pipeline_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert UCI HAR data to format compatible with our pipeline.

        Args:
            df: UCI HAR DataFrame

        Returns:
            DataFrame in pipeline-compatible format
        """
        logger.info("Converting UCI HAR data to pipeline format...")

        # The UCI HAR dataset doesn't contain raw accelerometer time series,
        # but we can extract some key features and simulate time series structure

        # Map UCI HAR features to our expected format
        pipeline_data = pd.DataFrame()

        # Basic info
        pipeline_data['user_id'] = df['Subject']
        pipeline_data['activity'] = df['Activity']
        pipeline_data['dataset_split'] = df.get('dataset_split', 'unknown')

        # Try to extract accelerometer-like features from UCI HAR features
        # UCI HAR features are already engineered, so we'll use some key ones
        acc_features = [col for col in df.columns if 'Acc' in col and any(axis in col for axis in ['X', 'Y', 'Z'])]

        if acc_features:
            # Use some representative accelerometer features
            x_features = [col for col in acc_features if 'X' in col]
            y_features = [col for col in acc_features if 'Y' in col]
            z_features = [col for col in acc_features if 'Z' in col]

            if x_features:
                pipeline_data['acc_x'] = df[x_features[0]]  # Use first X feature as representative
            if y_features:
                pipeline_data['acc_y'] = df[y_features[0]]  # Use first Y feature as representative  
            if z_features:
                pipeline_data['acc_z'] = df[z_features[0]]  # Use first Z feature as representative

        # If no accelerometer features found, create synthetic ones based on activity
        if not any(col in pipeline_data.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            logger.info("No direct accelerometer features found, creating synthetic time series...")
            pipeline_data = self._create_synthetic_timeseries(df)

        # Add timestamp (synthetic)
        pipeline_data['timestamp'] = np.arange(len(pipeline_data)) * (1 / 50)  # 50 Hz

        logger.info(f"Converted UCI HAR data: {pipeline_data.shape}")
        logger.info(f"Columns: {pipeline_data.columns.tolist()}")

        return pipeline_data

    def _create_synthetic_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic time series data based on UCI HAR activities.

        Args:
            df: Original UCI HAR DataFrame

        Returns:
            DataFrame with synthetic time series
        """
        logger.info("Creating synthetic time series from UCI HAR activities...")

        # Activity-based synthetic patterns
        activity_patterns = {
            'WALKING': {'freq': 1.5, 'amp_x': 0.5, 'amp_y': 0.3, 'amp_z': 0.2, 'noise': 0.1},
            'WALKING_UPSTAIRS': {'freq': 1.8, 'amp_x': 0.7, 'amp_y': 0.4, 'amp_z': 0.3, 'noise': 0.15},
            'WALKING_DOWNSTAIRS': {'freq': 1.3, 'amp_x': 0.6, 'amp_y': 0.4, 'amp_z': 0.25, 'noise': 0.12},
            'SITTING': {'freq': 0.0, 'amp_x': 0.0, 'amp_y': 0.0, 'amp_z': 0.0, 'noise': 0.05},
            'STANDING': {'freq': 0.1, 'amp_x': 0.1, 'amp_y': 0.1, 'amp_z': 0.1, 'noise': 0.08},
            'LAYING': {'freq': 0.05, 'amp_x': 0.05, 'amp_y': 0.05, 'amp_z': 0.05, 'noise': 0.03}
        }

        synthetic_data = []

        for idx, row in df.iterrows():
            activity = row['Activity']
            subject = row['Subject']

            if activity in activity_patterns:
                params = activity_patterns[activity]

                # Generate time series for this window (128 samples)
                n_samples = 128
                time_points = np.linspace(0, 2.56, n_samples)  # 2.56 seconds

                for t in time_points:
                    if params['freq'] > 0:
                        acc_x = (np.sin(2 * np.pi * params['freq'] * t) * params['amp_x'] +
                                 np.random.normal(0, params['noise']))
                        acc_y = (np.cos(2 * np.pi * params['freq'] * t) * params['amp_y'] +
                                 np.random.normal(0, params['noise']))
                        acc_z = (9.8 + np.sin(2 * np.pi * params['freq'] * 2 * t) * params['amp_z'] +
                                 np.random.normal(0, params['noise']))
                    else:
                        acc_x = np.random.normal(0, params['noise'])
                        acc_y = np.random.normal(0, params['noise'])
                        acc_z = 9.8 + np.random.normal(0, params['noise'])

                    synthetic_data.append({
                        'user_id': subject,
                        'activity': activity,
                        'acc_x': acc_x,
                        'acc_y': acc_y,
                        'acc_z': acc_z,
                        'window_id': idx,
                        'dataset_split': row.get('dataset_split', 'unknown')
                    })

        result_df = pd.DataFrame(synthetic_data)
        logger.info(f"Created synthetic time series: {result_df.shape}")

        return result_df

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the UCI HAR dataset.

        Returns:
            Dictionary with dataset information
        """
        return {
            'name': 'UCI Human Activity Recognition',
            'description': 'Human Activity Recognition database built from smartphone recordings',
            'activities': list(self.activity_mapping.values()),
            'n_activities': len(self.activity_mapping),
            'n_subjects': 30,
            'sampling_rate': self.sensor_info['sampling_rate'],
            'window_size': self.sensor_info['window_size'],
            'n_samples_per_window': self.sensor_info['n_samples_per_window'],
            'sensors': ['accelerometer', 'gyroscope'],
            'url': 'https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones',
            'citation': 'Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. '
                        'Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly '
                        'Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). '
                        'Vitoria-Gasteiz, Spain. Dec 2012'
        }


# Convenience function for backward compatibility
def load_uci_har_dataset(path: str) -> pd.DataFrame:
    """
    Load UCI HAR dataset (backward compatible function).

    Args:
        path: Path to UCI HAR dataset

    Returns:
        DataFrame with UCI HAR data
    """
    loader = UCIHARLoader()
    return loader.load_uci_har_dataset(path)


# Enhanced function with more options
def load_uci_har_for_pipeline(data_dir: str = "data/raw/UCI_HAR",
                              auto_download: bool = True,
                              convert_format: bool = True,
                              include_raw_signals: bool = False) -> pd.DataFrame:
    """
    Load UCI HAR dataset optimized for the wearable analysis pipeline.

    Args:
        data_dir: Directory to store/load UCI HAR dataset
        auto_download: Automatically download if not found
        convert_format: Convert to pipeline-compatible format
        include_raw_signals: Include raw sensor signals

    Returns:
        DataFrame ready for pipeline processing
    """
    loader = UCIHARLoader(data_dir)

    # Download if needed
    if auto_download and not loader.extracted_dir.exists():
        if not loader.download_dataset():
            raise RuntimeError("Failed to download UCI HAR dataset")

    # Load dataset
    df = loader.load_uci_har_dataset(include_raw_signals=include_raw_signals)

    # Convert format if requested
    if convert_format:
        df = loader.convert_to_pipeline_format(df)

    return df