"""
Feature Engineering Module

Extracts meaningful features from time-series wearable sensor data for machine learning.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Class for extracting features from wearable sensor time-series data.

    Supports various time-domain, frequency-domain, and statistical features.
    """

    def __init__(self, window_size: int = 25, overlap: float = 0.5, sampling_rate: float = 25.0):
        """
        Initialize FeatureEngineer.

        Args:
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0-1)
            sampling_rate: Sampling frequency in Hz
        """
        self.window_size = window_size
        self.overlap = max(0.0, min(0.99, overlap))  # Ensure overlap is between 0 and 0.99
        self.sampling_rate = sampling_rate
        self.step_size = max(1, int(window_size * (1 - overlap)))

        # Feature names will be stored here
        self.feature_names = []

        logger.info(f"FeatureEngineer initialized with window_size={window_size}, "
                   f"overlap={overlap}, step_size={self.step_size}, sampling_rate={sampling_rate}Hz")

    def extract_features(self, df: pd.DataFrame, group_by: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features from sensor data using sliding window approach.

        Args:
            df: DataFrame with sensor data
            group_by: Column to group by (e.g., 'user_id', 'session_id')

        Returns:
            Tuple of (features_df, labels_list)
        """
        logger.info("Starting feature extraction...")

        # Debug: Print dataset info
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")

        # Validate required columns
        required_cols = ['acc_x', 'acc_y', 'acc_z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. "
                           f"Available columns: {df.columns.tolist()}")

        # Check for valid data
        if len(df) == 0:
            raise ValueError("Input DataFrame is empty")

        if len(df) < self.window_size:
            logger.warning(f"Dataset has only {len(df)} samples, which is less than "
                         f"window_size ({self.window_size}). Reducing window size to {len(df)//2}")
            self.window_size = max(5, len(df) // 2)
            self.step_size = max(1, int(self.window_size * (1 - self.overlap)))

        features_list = []
        labels_list = []

        if group_by and group_by in df.columns:
            # Process each group separately
            groups = df.groupby(group_by)
            total_groups = len(groups)

            logger.info(f"Processing {total_groups} groups by {group_by}")

            for group_name, group_df in tqdm(groups, desc="Processing groups"):
                if len(group_df) >= self.window_size:
                    group_features, group_labels = self._extract_windows_from_data(group_df)
                    features_list.extend(group_features)
                    labels_list.extend(group_labels)
                else:
                    logger.debug(f"Skipping group {group_name}: only {len(group_df)} samples")
        else:
            # Process entire dataset
            features_list, labels_list = self._extract_windows_from_data(df)

        # Check if any features were extracted
        if not features_list:
            logger.error("No features were extracted!")
            logger.error(f"Dataset info: shape={df.shape}, window_size={self.window_size}, "
                        f"step_size={self.step_size}")

            # Try with smaller window size
            if self.window_size > 10:
                logger.info("Trying with smaller window size...")
                original_window_size = self.window_size
                self.window_size = 10
                self.step_size = max(1, int(self.window_size * (1 - self.overlap)))

                features_list, labels_list = self._extract_windows_from_data(df)

                if features_list:
                    logger.warning(f"Successfully extracted features with reduced window size "
                                 f"({self.window_size} instead of {original_window_size})")
                else:
                    logger.error("Still no features extracted even with smaller window size")
                    # Provide debugging information
                    logger.error(f"Sample data (first 5 rows):\n{df.head()}")
                    raise ValueError("Feature extraction failed. Check your data format and parameters.")

        # Convert to DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)

            # Store feature names
            self.feature_names = features_df.columns.tolist()

            logger.info(f"Extracted {len(features_df)} feature windows with {len(features_df.columns)} features each")
            logger.info(f"Label distribution: {pd.Series(labels_list).value_counts().to_dict()}")

            return features_df, labels_list
        else:
            raise ValueError("No features could be extracted from the data. "
                           "Check that your data contains valid accelerometer readings.")


    def _extract_windows_from_data(self, df: pd.DataFrame) -> Tuple[List[Dict], List[str]]:
        """
        Extract features from windows of a single dataset.

        Args:
            df: DataFrame with sensor data

        Returns:
            Tuple of (features_list, labels_list)
        """
        features_list = []
        labels_list = []

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate number of windows more permissively
        max_possible_windows = (len(df) - self.window_size) // self.step_size + 1
        n_windows = max(1, max_possible_windows)

        logger.info(f"Processing {len(df)} samples into up to {n_windows} windows "
                   f"(window_size={self.window_size}, step_size={self.step_size})")

        successful_extractions = 0

        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = min(start_idx + self.window_size, len(df))

            # Extract window
            window_data = df.iloc[start_idx:end_idx]

            # Very permissive window size check - accept any window with at least 3 samples
            min_window_size = max(3, int(self.window_size * 0.3))
            if len(window_data) < min_window_size:
                logger.debug(f"Skipping window {i}: size {len(window_data)} < {min_window_size}")
                continue

            # Determine dominant activity (if activity column exists)
            if 'activity' in window_data.columns:
                activity_counts = window_data['activity'].value_counts()

                # Very permissive activity purity - just take the most common activity
                # No minimum purity threshold - always accept the dominant activity
                dominant_activity = activity_counts.index[0]

                # Only skip if the window is completely empty somehow
                if len(activity_counts) == 0:
                    logger.debug(f"Skipping window {i}: no activities found")
                    continue
            else:
                dominant_activity = 'unknown'

            # Extract features for this window
            try:
                features = self._extract_window_features(window_data)
                if features and len(features) > 0:  # Only add if features were successfully extracted
                    features_list.append(features)
                    labels_list.append(dominant_activity)
                    successful_extractions += 1

                    if successful_extractions % 100 == 0:  # Progress logging
                        logger.debug(f"Processed {successful_extractions} windows...")
            except Exception as e:
                logger.debug(f"Failed to extract features for window {i}: {str(e)}")
                continue

        logger.info(f"Successfully extracted {len(features_list)} feature windows")

        # If we still got very few windows, try with much smaller step size
        if len(features_list) < 20 and self.step_size > 1:
            logger.warning(f"Only got {len(features_list)} windows, trying step_size=1...")
            original_step_size = self.step_size
            self.step_size = 1  # Use step size of 1 for maximum windows

            # Retry with step size of 1
            features_list_retry, labels_list_retry = self._extract_windows_from_data(df)

            if len(features_list_retry) > len(features_list):
                logger.info(f"Improved from {len(features_list)} to {len(features_list_retry)} windows "
                           f"with step_size=1")
                return features_list_retry, labels_list_retry
            else:
                # Restore original step size
                self.step_size = original_step_size

        return features_list, labels_list

    def _extract_window_features(self, window_data: pd.DataFrame) -> Dict:
        """
        Extract all features from a single window of data.

        Args:
            window_data: DataFrame with one window of sensor data

        Returns:
            Dictionary with extracted features
        """
        features = {}

        # Define sensor columns to process
        sensor_columns = ['acc_x', 'acc_y', 'acc_z']

        # Add derived columns if they exist
        derived_columns = ['acc_magnitude', 'acc_total_no_gravity']
        for col in derived_columns:
            if col in window_data.columns:
                sensor_columns.append(col)

        # Extract features for each sensor column
        for col in sensor_columns:
            if col in window_data.columns:
                values = window_data[col].values
                col_features = self._extract_single_signal_features(values, col)
                features.update(col_features)

        # Extract cross-axis features
        if all(col in window_data.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            cross_features = self._extract_cross_axis_features(window_data)
            features.update(cross_features)

        return features

    def _extract_single_signal_features(self, values: np.ndarray, prefix: str) -> Dict:
        """
        Extract features from a single signal.

        Args:
            values: 1D array of signal values
            prefix: Prefix for feature names

        Returns:
            Dictionary with features
        """
        features = {}

        # Basic statistical features
        features[f'{prefix}_mean'] = np.mean(values)
        features[f'{prefix}_std'] = np.std(values)
        features[f'{prefix}_var'] = np.var(values)
        features[f'{prefix}_min'] = np.min(values)
        features[f'{prefix}_max'] = np.max(values)
        features[f'{prefix}_range'] = np.max(values) - np.min(values)
        features[f'{prefix}_median'] = np.median(values)

        # Percentile features
        features[f'{prefix}_q25'] = np.percentile(values, 25)
        features[f'{prefix}_q75'] = np.percentile(values, 75)
        features[f'{prefix}_iqr'] = features[f'{prefix}_q75'] - features[f'{prefix}_q25']

        # Root Mean Square
        features[f'{prefix}_rms'] = np.sqrt(np.mean(values**2))

        # Shape features
        try:
            features[f'{prefix}_skewness'] = stats.skew(values)
            features[f'{prefix}_kurtosis'] = stats.kurtosis(values)
        except:
            features[f'{prefix}_skewness'] = 0
            features[f'{prefix}_kurtosis'] = 0

        # Energy features
        features[f'{prefix}_energy'] = np.sum(values**2)
        features[f'{prefix}_signal_power'] = np.mean(values**2)

        # Peak detection features
        try:
            peaks, properties = find_peaks(values, height=np.mean(values))
            features[f'{prefix}_num_peaks'] = len(peaks)
            features[f'{prefix}_peak_frequency'] = len(peaks) / (self.window_size / self.sampling_rate)

            if len(peaks) > 0:
                features[f'{prefix}_avg_peak_height'] = np.mean(properties['peak_heights'])
                features[f'{prefix}_max_peak_height'] = np.max(properties['peak_heights'])
            else:
                features[f'{prefix}_avg_peak_height'] = 0
                features[f'{prefix}_max_peak_height'] = 0
        except:
            features[f'{prefix}_num_peaks'] = 0
            features[f'{prefix}_peak_frequency'] = 0
            features[f'{prefix}_avg_peak_height'] = 0
            features[f'{prefix}_max_peak_height'] = 0

        # Zero crossing features
        mean_centered = values - np.mean(values)
        zero_crossings = np.where(np.diff(np.signbit(mean_centered)))[0]
        features[f'{prefix}_zero_crossings'] = len(zero_crossings)
        features[f'{prefix}_zero_crossing_rate'] = len(zero_crossings) / len(values)

        # Frequency domain features
        try:
            fft = np.fft.fft(values)
            fft_magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(values), 1/self.sampling_rate)

            # Only consider positive frequencies
            positive_freq_idx = freqs > 0
            positive_freqs = freqs[positive_freq_idx]
            positive_magnitude = fft_magnitude[positive_freq_idx]

            if len(positive_magnitude) > 0:
                # Dominant frequency
                dominant_freq_idx = np.argmax(positive_magnitude)
                features[f'{prefix}_dominant_freq'] = positive_freqs[dominant_freq_idx]
                features[f'{prefix}_dominant_freq_magnitude'] = positive_magnitude[dominant_freq_idx]

                # Spectral features
                features[f'{prefix}_spectral_energy'] = np.sum(positive_magnitude**2)
                features[f'{prefix}_spectral_entropy'] = self._calculate_spectral_entropy(positive_magnitude)

                # Frequency band energies (0-1Hz, 1-5Hz, 5-12.5Hz)
                freq_bands = [(0, 1), (1, 5), (5, self.sampling_rate/2)]
                for i, (low, high) in enumerate(freq_bands):
                    band_mask = (positive_freqs >= low) & (positive_freqs < high)
                    band_energy = np.sum(positive_magnitude[band_mask]**2)
                    features[f'{prefix}_band_{i}_energy'] = band_energy
            else:
                features[f'{prefix}_dominant_freq'] = 0
                features[f'{prefix}_dominant_freq_magnitude'] = 0
                features[f'{prefix}_spectral_energy'] = 0
                features[f'{prefix}_spectral_entropy'] = 0
                for i in range(3):
                    features[f'{prefix}_band_{i}_energy'] = 0
        except:
            # Fallback values for frequency features
            features[f'{prefix}_dominant_freq'] = 0
            features[f'{prefix}_dominant_freq_magnitude'] = 0
            features[f'{prefix}_spectral_energy'] = 0
            features[f'{prefix}_spectral_entropy'] = 0
            for i in range(3):
                features[f'{prefix}_band_{i}_energy'] = 0

        return features

    def _extract_cross_axis_features(self, window_data: pd.DataFrame) -> Dict:
        """
        Extract features that involve multiple axes.

        Args:
            window_data: DataFrame with sensor data

        Returns:
            Dictionary with cross-axis features
        """
        features = {}

        # Correlation features
        try:
            # Check if we have enough variance for correlation
            acc_cols = ['acc_x', 'acc_y', 'acc_z']
            valid_correlations = True

            for col in acc_cols:
                if col in window_data.columns:
                    if window_data[col].std() < 1e-8:  # Very low variance
                        valid_correlations = False
                        break

            if valid_correlations:
                corr_xy = np.corrcoef(window_data['acc_x'], window_data['acc_y'])[0, 1]
                corr_xz = np.corrcoef(window_data['acc_x'], window_data['acc_z'])[0, 1]
                corr_yz = np.corrcoef(window_data['acc_y'], window_data['acc_z'])[0, 1]

                # Handle NaN correlations
                features['acc_xy_correlation'] = 0 if np.isnan(corr_xy) else corr_xy
                features['acc_xz_correlation'] = 0 if np.isnan(corr_xz) else corr_xz
                features['acc_yz_correlation'] = 0 if np.isnan(corr_yz) else corr_yz
            else:
                # Use default values when correlation cannot be computed
                features['acc_xy_correlation'] = 0
                features['acc_xz_correlation'] = 0
                features['acc_yz_correlation'] = 0
        except Exception as e:
            # Fallback to zero correlations
            features['acc_xy_correlation'] = 0
            features['acc_xz_correlation'] = 0
            features['acc_yz_correlation'] = 0

        # Signal vector magnitude statistics
        if 'acc_magnitude' in window_data.columns:
            svm = window_data['acc_magnitude'].values
            features['svm_mean'] = np.mean(svm)
            features['svm_std'] = np.std(svm)
            features['svm_range'] = np.max(svm) - np.min(svm)
        else:
            # Calculate magnitude if not present
            if all(col in window_data.columns for col in ['acc_x', 'acc_y', 'acc_z']):
                svm = np.sqrt(window_data['acc_x']**2 + window_data['acc_y']**2 + window_data['acc_z']**2)
                features['svm_mean'] = np.mean(svm)
                features['svm_std'] = np.std(svm)
                features['svm_range'] = np.max(svm) - np.min(svm)

        # Tilt features (if we assume z is vertical)
        try:
            # Average tilt from vertical
            avg_acc_z = np.mean(window_data['acc_z'])
            # Clamp to valid range for arccos
            normalized_z = np.clip(avg_acc_z / 9.8, -1, 1)
            features['avg_tilt_angle'] = np.arccos(normalized_z) * 180 / np.pi
        except Exception:
            features['avg_tilt_angle'] = 0

        return features

    def _calculate_spectral_entropy(self, magnitude_spectrum: np.ndarray) -> float:
        """
        Calculate spectral entropy of a signal.

        Args:
            magnitude_spectrum: FFT magnitude spectrum

        Returns:
            Spectral entropy value
        """
        # Normalize to get probability distribution
        power_spectrum = magnitude_spectrum**2
        total_power = np.sum(power_spectrum)

        if total_power == 0:
            return 0

        prob_spectrum = power_spectrum / total_power

        # Remove zeros to avoid log(0)
        prob_spectrum = prob_spectrum[prob_spectrum > 0]

        if len(prob_spectrum) == 0:
            return 0

        # Calculate entropy
        entropy = -np.sum(prob_spectrum * np.log2(prob_spectrum))
        return entropy

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.

        Returns:
            List of feature names
        """
        return self.feature_names.copy()

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.

        Returns:
            Dictionary mapping feature types to feature names
        """
        if not self.feature_names:
            return {}

        groups = {
            'statistical': [],
            'time_domain': [],
            'frequency_domain': [],
            'peak_detection': [],
            'cross_axis': [],
            'energy': []
        }

        for name in self.feature_names:
            if any(keyword in name for keyword in ['mean', 'std', 'var', 'min', 'max', 'median', 'q25', 'q75', 'iqr']):
                groups['statistical'].append(name)
            elif any(keyword in name for keyword in ['skewness', 'kurtosis', 'rms', 'zero_crossing']):
                groups['time_domain'].append(name)
            elif any(keyword in name for keyword in ['freq', 'spectral', 'band']):
                groups['frequency_domain'].append(name)
            elif any(keyword in name for keyword in ['peak', 'num_peaks']):
                groups['peak_detection'].append(name)
            elif any(keyword in name for keyword in ['correlation', 'svm', 'tilt']):
                groups['cross_axis'].append(name)
            elif any(keyword in name for keyword in ['energy', 'power']):
                groups['energy'].append(name)

        return groups

    def save_features(self, features_df: pd.DataFrame, labels: List[str], filepath: str) -> None:
        """
        Save extracted features to file.

        Args:
            features_df: DataFrame with features
            labels: List of activity labels
            filepath: Path to save file
        """
        # Combine features and labels
        data_to_save = features_df.copy()
        data_to_save['activity'] = labels

        # Save to CSV
        data_to_save.to_csv(filepath, index=False)
        logger.info(f"Saved {len(features_df)} feature samples to {filepath}")

    def load_features(self, filepath: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load previously extracted features.

        Args:
            filepath: Path to feature file

        Returns:
            Tuple of (features_df, labels)
        """
        data = pd.read_csv(filepath)

        if 'activity' not in data.columns:
            raise ValueError("Feature file must contain 'activity' column")

        features_df = data.drop('activity', axis=1)
        labels = data['activity'].tolist()

        # Update feature names
        self.feature_names = features_df.columns.tolist()

        logger.info(f"Loaded {len(features_df)} feature samples from {filepath}")
        return features_df, labels

    def analyze_feature_importance(self, features_df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
        """
        Analyze feature importance using correlation and variance.

        Args:
            features_df: DataFrame with features
            labels: List of activity labels

        Returns:
            DataFrame with feature analysis
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_selection import mutual_info_classif

        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        analysis = []

        for col in features_df.columns:
            feature_data = features_df[col]

            # Basic statistics
            variance = feature_data.var()
            mean_val = feature_data.mean()
            std_val = feature_data.std()

            # Mutual information with target
            try:
                mi_score = mutual_info_classif(feature_data.values.reshape(-1, 1),
                                             labels_encoded,
                                             random_state=42)[0]
            except:
                mi_score = 0

            analysis.append({
                'feature': col,
                'variance': variance,
                'mean': mean_val,
                'std': std_val,
                'mutual_info': mi_score,
                'coefficient_of_variation': std_val / abs(mean_val) if mean_val != 0 else float('inf')
            })

        analysis_df = pd.DataFrame(analysis)
        analysis_df = analysis_df.sort_values('mutual_info', ascending=False)

        logger.info("Feature importance analysis completed")
        return analysis_df

    def select_top_features(self, features_df: pd.DataFrame, labels: List[str],
                           n_features: int = 50, method: str = 'mutual_info') -> List[str]:
        """
        Select top features based on importance criteria.

        Args:
            features_df: DataFrame with features
            labels: List of activity labels
            n_features: Number of features to select
            method: Selection method ('mutual_info', 'variance', 'correlation')

        Returns:
            List of selected feature names
        """
        if method == 'mutual_info':
            analysis_df = self.analyze_feature_importance(features_df, labels)
            selected_features = analysis_df.head(n_features)['feature'].tolist()

        elif method == 'variance':
            # Select features with highest variance
            variances = features_df.var().sort_values(ascending=False)
            selected_features = variances.head(n_features).index.tolist()

        elif method == 'correlation':
            # Select features with low correlation to each other
            corr_matrix = features_df.corr().abs()
            selected_features = []
            remaining_features = list(features_df.columns)

            while len(selected_features) < n_features and remaining_features:
                # Select feature with highest variance among remaining
                candidate = features_df[remaining_features].var().idxmax()
                selected_features.append(candidate)
                remaining_features.remove(candidate)

                # Remove highly correlated features
                if candidate in corr_matrix.columns:
                    correlated = corr_matrix[candidate][corr_matrix[candidate] > 0.9].index
                    remaining_features = [f for f in remaining_features if f not in correlated]

        else:
            raise ValueError(f"Unknown selection method: {method}")

        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features