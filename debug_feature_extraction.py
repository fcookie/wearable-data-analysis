#!/usr/bin/env python3
"""
Debug script to test feature extraction with small dataset
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
import pandas as pd


def debug_feature_extraction():
    print("🔍 Debugging Feature Extraction...")

    # Create small test dataset
    data_loader = DataLoader()
    print("\n1. Generating small test dataset...")
    raw_data = data_loader.generate_sample_data(n_samples=100)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data columns: {raw_data.columns.tolist()}")
    print(f"Sample data:\n{raw_data.head()}")

    # Preprocess
    print("\n2. Preprocessing...")
    processed_data = data_loader.preprocess_data(raw_data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed columns: {processed_data.columns.tolist()}")

    # Test different window sizes
    window_sizes = [5, 10, 25]

    for window_size in window_sizes:
        print(f"\n3. Testing window_size={window_size}...")

        feature_engineer = FeatureEngineer(
            window_size=window_size,
            overlap=0.5,
            sampling_rate=25.0
        )

        print(f"   Window size: {feature_engineer.window_size}")
        print(f"   Step size: {feature_engineer.step_size}")
        print(f"   Data length: {len(processed_data)}")

        # Calculate expected windows
        if len(processed_data) >= window_size:
            expected_windows = max(1, (len(processed_data) - window_size) // feature_engineer.step_size + 1)
            print(f"   Expected windows: {expected_windows}")

            try:
                features_df, labels = feature_engineer.extract_features(processed_data)
                print(f"   ✅ Success! Extracted {len(features_df)} windows with {features_df.shape[1]} features")
                print(f"   Labels: {set(labels)}")
                break
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
        else:
            print(f"   ⚠️ Skipped: data too small ({len(processed_data)} < {window_size})")

    print("\n🎉 Debug complete!")


if __name__ == "__main__":
    debug_feature_extraction()