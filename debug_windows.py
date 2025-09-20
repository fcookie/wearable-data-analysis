#!/usr/bin/env python3
"""
Debug window extraction to see why so few windows are being created
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from data_loader import DataLoader
from feature_engineer import FeatureEngineer


def debug_windows():
    print("🔍 Debugging Window Extraction")
    print("=" * 50)

    # Generate small dataset
    data_loader = DataLoader()
    raw_data = data_loader.generate_sample_data(n_samples=200)
    processed_data = data_loader.preprocess_data(raw_data)

    print(f"Data shape: {processed_data.shape}")
    print(f"Activities: {processed_data['activity'].value_counts().to_dict()}")

    # Test different parameters
    configs = [
        {'window_size': 5, 'overlap': 0.0},
        {'window_size': 5, 'overlap': 0.5},
        {'window_size': 10, 'overlap': 0.0},
        {'window_size': 10, 'overlap': 0.3},
        {'window_size': 25, 'overlap': 0.5},
    ]

    for config in configs:
        print(f"\n--- Testing: window_size={config['window_size']}, overlap={config['overlap']} ---")

        fe = FeatureEngineer(**config)
        step_size = fe.step_size

        print(f"Step size: {step_size}")
        print(f"Expected max windows: {(len(processed_data) - config['window_size']) // step_size + 1}")

        # Manual window inspection
        data_len = len(processed_data)
        window_size = config['window_size']

        valid_windows = 0
        rejected_size = 0
        rejected_purity = 0

        for i in range(0, data_len - window_size + 1, step_size):
            window_data = processed_data.iloc[i:i + window_size]

            # Check size
            if len(window_data) < window_size * 0.3:
                rejected_size += 1
                continue

            # Check activity purity
            activity_counts = window_data['activity'].value_counts()
            dominant_count = activity_counts.iloc[0]
            total_count = len(window_data)

            # Check if it would pass the old strict criteria
            old_threshold = max(1, int(total_count * 0.7))
            if len(activity_counts) > 1 and dominant_count < old_threshold:
                rejected_purity += 1
                print(
                    f"  Window {i // step_size}: would reject for purity ({dominant_count}/{total_count} = {dominant_count / total_count:.1%})")
                continue

            valid_windows += 1

        print(f"Manual analysis:")
        print(f"  Valid windows: {valid_windows}")
        print(f"  Rejected for size: {rejected_size}")
        print(f"  Rejected for purity: {rejected_purity}")

        # Now test actual extraction
        try:
            features_df, labels = fe.extract_features(processed_data)
            print(f"  Actual extracted: {len(features_df)}")
            if len(labels) > 0:
                print(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        except Exception as e:
            print(f"  Extraction failed: {e}")


if __name__ == "__main__":
    debug_windows()