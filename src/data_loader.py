"""
Data Loader Module

Handles loading, preprocessing, and initial cleaning of wearable sensor data.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and preprocessing wearable sensor data.

    Supports multiple data formats and provides consistent preprocessing
    for accelerometer and other sensor data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")

        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Supported file formats
        self.supported_formats = ['.csv', '.txt', '.json']

    def load_csv_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {str(e)}")
            raise

    def load_multiple_files(self, file_pattern: str, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load and combine multiple data files.

        Args:
            file_pattern: Pattern to match files (e.g., "*.csv")
            data_dir: Directory to search (defaults to raw_data_dir)

        Returns:
            Combined DataFrame
        """
        import glob

        if data_dir is None:
            data_dir = self.raw_data_dir

        file_path = os.path.join(data_dir, file_pattern)
        files = glob.glob(file_path)

        if not files:
            logger.warning(f"No files found matching pattern: {file_path}")
            return pd.DataFrame()

        dataframes = []
        for file in files:
            try:
                df = self.load_csv_data(file)
                df['source_file'] = os.path.basename(file)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {str(e)}")
                continue

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined {len(files)} files into DataFrame with {len(combined_df)} records")
            return combined_df
        else:
            logger.error("No files could be loaded successfully")
            return pd.DataFrame()

    def generate_sample_data(self, n_samples: int = 10000, sampling_rate: float = 25.0) -> pd.DataFrame:
        """
        Generate synthetic wearable sensor data for testing/demonstration.

        Args:
            n_samples: Number of data points to generate
            sampling_rate: Sampling frequency in Hz

        Returns:
            DataFrame with synthetic sensor data
        """
        logger.info(f"Generating {n_samples} synthetic data points at {sampling_rate}Hz")

        # Define activities and their characteristics
        activities = {
            'walking': {'freq': 1.5, 'amp_x': 0.5, 'amp_y': 0.3, 'amp_z': 0.2, 'noise': 0.1},
            'running': {'freq': 3.0, 'amp_x': 1.2, 'amp_y': 0.8, 'amp_z': 0.5, 'noise': 0.2},
            'sitting': {'freq': 0.0, 'amp_x': 0.0, 'amp_y': 0.0, 'amp_z': 0.0, 'noise': 0.05},
            'standing': {'freq': 0.1, 'amp_x': 0.1, 'amp_y': 0.1, 'amp_z': 0.1, 'noise': 0.1},
            'climbing_stairs': {'freq': 2.0, 'amp_x': 0.8, 'amp_y': 0.6, 'amp_z': 0.4, 'noise': 0.3}
        }

        # Generate time series
        time_interval = 1.0 / sampling_rate
        time = np.arange(0, n_samples) * time_interval

        data = []
        np.random.seed(42)  # For reproducibility

        # Generate activities in longer sequences to reduce mixed windows
        activity_list = list(activities.keys())
        current_activity = np.random.choice(activity_list)
        activity_duration = np.random.randint(50, 200)  # 50-200 samples per activity sequence
        samples_in_current_activity = 0

        for i in range(n_samples):
            # Switch activity if we've been in current activity long enough
            if samples_in_current_activity >= activity_duration:
                current_activity = np.random.choice(activity_list)
                activity_duration = np.random.randint(50, 200)  # New duration
                samples_in_current_activity = 0

            params = activities[current_activity]

            # Generate accelerometer data based on activity
            if params['freq'] > 0:
                acc_x = (np.sin(2 * np.pi * params['freq'] * time[i]) * params['amp_x'] +
                        np.random.normal(0, params['noise']))
                acc_y = (np.cos(2 * np.pi * params['freq'] * time[i]) * params['amp_y'] +
                        np.random.normal(0, params['noise']))
                acc_z = (9.8 + np.sin(2 * np.pi * params['freq'] * 2 * time[i]) * params['amp_z'] +
                        np.random.normal(0, params['noise']))
            else:
                acc_x = np.random.normal(0, params['noise'])
                acc_y = np.random.normal(0, params['noise'])
                acc_z = 9.8 + np.random.normal(0, params['noise'])

            data.append({
                'timestamp': time[i],
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'activity': current_activity,
                'user_id': np.random.randint(1, 6),  # 5 synthetic users
                'session_id': np.random.randint(1, 4)  # 3 sessions per user
            })

            samples_in_current_activity += 1

        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic data with shape: {df.shape}")
        logger.info(f"Activities distribution: {df['activity'].value_counts().to_dict()}")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw sensor data.

        Args:
            df: Raw sensor DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        original_size = len(df)

        # Create a copy to avoid modifying original data
        df_processed = df.copy()

        # Handle missing values
        missing_before = df_processed.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Found {missing_before} missing values")

            # Drop rows with missing accelerometer data
            critical_columns = ['acc_x', 'acc_y', 'acc_z']
            df_processed = df_processed.dropna(subset=critical_columns)

            # Fill missing activity labels with 'unknown'
            if 'activity' in df_processed.columns:
                df_processed['activity'] = df_processed['activity'].fillna('unknown')

        # Remove outliers (values beyond 3 standard deviations)
        for col in ['acc_x', 'acc_y', 'acc_z']:
            if col in df_processed.columns:
                mean_val = df_processed[col].mean()
                std_val = df_processed[col].std()
                threshold = 3 * std_val

                outliers_mask = np.abs(df_processed[col] - mean_val) > threshold
                outliers_count = outliers_mask.sum()

                if outliers_count > 0:
                    logger.info(f"Removing {outliers_count} outliers from {col}")
                    df_processed = df_processed[~outliers_mask]

        # Calculate derived features
        self._add_derived_features(df_processed)

        # Sort by timestamp if available
        if 'timestamp' in df_processed.columns:
            df_processed = df_processed.sort_values('timestamp').reset_index(drop=True)

        final_size = len(df_processed)
        removed_records = original_size - final_size

        logger.info(f"Preprocessing complete. Removed {removed_records} records "
                   f"({removed_records/original_size*100:.1f}%)")
        logger.info(f"Final dataset shape: {df_processed.shape}")

        return df_processed

    def _add_derived_features(self, df: pd.DataFrame) -> None:
        """
        Add derived features to the DataFrame (in-place).

        Args:
            df: DataFrame to modify
        """
        # Calculate acceleration magnitude
        if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

            # Remove gravity from z-axis (assuming device orientation)
            df['acc_z_no_gravity'] = df['acc_z'] - 9.8

            # Total acceleration without gravity
            df['acc_total_no_gravity'] = np.sqrt(
                df['acc_x']**2 + df['acc_y']**2 + df['acc_z_no_gravity']**2
            )

            logger.info("Added derived acceleration features")

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to file.

        Args:
            df: DataFrame to save
            filename: Name of the output file

        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.processed_data_dir, filename)

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {str(e)}")
            raise

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load previously processed data.

        Args:
            filename: Name of the processed data file

        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.processed_data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed data file not found: {filepath}")

        return self.load_csv_data(filepath)

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the dataset.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }

        # Activity distribution if available
        if 'activity' in df.columns:
            summary['activity_distribution'] = df['activity'].value_counts().to_dict()

        # User distribution if available
        if 'user_id' in df.columns:
            summary['user_distribution'] = df['user_id'].value_counts().to_dict()

        # Numeric column statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            summary['numeric_stats'] = df[numeric_columns].describe().to_dict()

        return summary

    def validate_data_format(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the data has the expected format.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check for required columns
        required_columns = ['acc_x', 'acc_y', 'acc_z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # Check data types
        for col in required_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} should be numeric")

        # Check for reasonable accelerometer values (-20g to +20g)
        for col in required_columns:
            if col in df.columns:
                if df[col].min() < -20 or df[col].max() > 20:
                    issues.append(f"Column {col} has unrealistic values (outside ±20g)")

        # Check for sufficient data
        if len(df) < 100:
            issues.append("Dataset too small (less than 100 samples)")

        is_valid = len(issues) == 0
        return is_valid, issues