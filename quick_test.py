#!/usr/bin/env python3
"""
Quick test to verify the pipeline works with smaller parameters
Run this from the project root directory where main.py is located.
"""

import sys
import os

# Add src directory to path (same as run_analysis.py)
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def quick_test():
    print("🧪 Quick Pipeline Test")
    print("=" * 40)
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Src path: {src_path}")
    print(f"Src exists: {os.path.exists(src_path)}")

    if os.path.exists(src_path):
        print(f"Files in src: {os.listdir(src_path)}")

    try:
        # Check if we can import
        print("\nAttempting imports...")
        from data_loader import DataLoader
        print("✅ DataLoader imported")
        from feature_engineer import FeatureEngineer
        print("✅ FeatureEngineer imported")
        from model import WearableModel
        print("✅ WearableModel imported")

        # Step 1: Generate small dataset
        print("\n1. Generating data...")
        data_loader = DataLoader()
        raw_data = data_loader.generate_sample_data(n_samples=1000)
        print(f"   Generated: {raw_data.shape}")

        # Step 2: Preprocess
        print("2. Preprocessing...")
        processed_data = data_loader.preprocess_data(raw_data)
        print(f"   Processed: {processed_data.shape}")

        # Step 3: Extract features with small window
        print("3. Feature extraction...")
        feature_engineer = FeatureEngineer(
            window_size=10,  # Small window
            overlap=0.1,  # Very little overlap to get more windows
            sampling_rate=25.0
        )

        features_df, labels = feature_engineer.extract_features(processed_data)
        print(f"   Features: {features_df.shape}")
        print(f"   Labels: {len(labels)} ({len(set(labels))} unique)")

        if len(features_df) < 5:
            print("⚠️ Too few feature windows. Trying with smaller window...")
            feature_engineer = FeatureEngineer(
                window_size=5,  # Even smaller window
                overlap=0.0,  # No overlap
                sampling_rate=25.0
            )
            features_df, labels = feature_engineer.extract_features(processed_data)
            print(f"   Retry - Features: {features_df.shape}")

        if len(features_df) == 0:
            print("❌ No features extracted!")
            return
        elif len(features_df) < 10:
            print("⚠️ Very few features extracted. Need more data or smaller windows.")
            print("🎯 Try: python run_analysis.py --generate-data --n-samples 10000 --window-size 5 --overlap 0.0")
            return

        # Step 4: Quick model test
        print("4. Training model...")
        model = WearableModel()

        # Adjust test size if we have few samples
        test_size = 0.1 if len(features_df) < 50 else 0.3
        cv_folds = 2 if len(features_df) < 30 else 3

        results = model.train(
            features_df,
            labels,
            test_size=test_size,
            cv_folds=cv_folds,
            models_to_train=['Random Forest']  # Just one model
        )

        best_accuracy = results[model.best_model_name]['test_accuracy']
        print(f"   Best accuracy: {best_accuracy:.3f}")

        print("\n✅ SUCCESS! Pipeline is working.")
        print(f"🎯 Try these parameters:")
        print(f"   python run_analysis.py --generate-data --window-size 10 --overlap 0.3")

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {str(e)}")
        print("\n💡 Make sure you're running this from the project root directory!")
        print("   cd wearable-data-analysis")
        print("   python quick_test.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()