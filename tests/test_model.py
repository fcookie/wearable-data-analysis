#!/usr/bin/env python3
"""
Unit tests for the WearableModel class.

Run with: python -m pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from model import WearableModel
from data_loader import DataLoader
from feature_engineer import FeatureEngineer


class TestWearableModel:
    """Test class for WearableModel functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Generate small synthetic dataset
        data_loader = DataLoader()
        raw_data = data_loader.generate_sample_data(n_samples=1000)
        processed_data = data_loader.preprocess_data(raw_data)

        # Extract features
        feature_engineer = FeatureEngineer(window_size=25, overlap=0.5)
        features_df, labels = feature_engineer.extract_features(processed_data)

        return features_df, labels

    @pytest.fixture
    def model(self):
        """Create a WearableModel instance."""
        return WearableModel(random_state=42)

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert not model.is_trained
        assert model.best_model is None
        assert len(model.model_configs) > 0

    def test_model_training(self, model, sample_data):
        """Test model training functionality."""
        features_df, labels = sample_data

        # Train models (use subset for faster testing)
        results = model.train(features_df, labels, models_to_train=['Random Forest'])

        assert len(results) > 0
        assert model.is_trained
        assert model.best_model is not None
        assert model.best_model_name is not None

        # Check if results contain required keys
        for model_name, result in results.items():
            required_keys = ['train_accuracy', 'test_accuracy', 'cv_scores', 'cv_mean', 'cv_std']
            for key in required_keys:
                assert key in result

    def test_model_prediction(self, model, sample_data):
        """Test model prediction functionality."""
        features_df, labels = sample_data

        # Train model first
        model.train(features_df, labels, models_to_train=['Random Forest'])

        # Test single prediction
        sample_features = features_df.iloc[0].values
        activity, confidence = model.predict_single(sample_features)

        assert isinstance(activity, str)
        assert 0 <= confidence <= 1

        # Test batch prediction
        batch_features = features_df.iloc[:5]
        activities, probabilities = model.predict(batch_features)

        assert len(activities) == 5
        assert probabilities.shape[0] == 5
        assert all(isinstance(act, str) for act in activities)

    def test_model_save_load(self, model, sample_data):
        """Test model saving and loading."""
        features_df, labels = sample_data

        # Train model
        model.train(features_df, labels, models_to_train=['Random Forest'])

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model.save_model(tmp_file.name)

            # Create new model instance and load
            new_model = WearableModel()
            new_model.load_model(tmp_file.name)

            assert new_model.is_trained
            assert new_model.best_model_name == model.best_model_name

            # Test prediction with loaded model
            sample_features = features_df.iloc[0].values
            activity1, conf1 = model.predict_single(sample_features)
            activity2, conf2 = new_model.predict_single(sample_features)

            assert activity1 == activity2
            assert abs(conf1 - conf2) < 0.001  # Should be very similar

            # Clean up
            os.unlink(tmp_file.name)

    def test_cross_validation(self, model, sample_data):
        """Test cross-validation functionality."""
        features_df, labels = sample_data

        # Prepare label encoder
        model.label_encoder.fit(labels)

        # Test cross-validation
        cv_results = model.cross_validate_model(features_df, labels, 'Random Forest', cv_folds=3)

        assert 'accuracy_mean' in cv_results
        assert 'accuracy_std' in cv_results
        assert 0 <= cv_results['accuracy_mean'] <= 1

    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        features_df, labels = sample_data

        # Train model with Random Forest (which has feature importance)
        model.train(features_df, labels, models_to_train=['Random Forest'])

        # Get feature importance
        importance_df = model.get_feature_importance(top_n=10)

        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)

    def test_model_summary(self, model, sample_data):
        """Test model summary generation."""
        features_df, labels = sample_data

        # Train models
        model.train(features_df, labels, models_to_train=['Random Forest', 'Logistic Regression'])

        # Get summary
        summary = model.get_model_summary()

        assert 'best_model' in summary
        assert len(summary) > 1  # Should have model results + best_model key

        for model_name in ['Random Forest', 'Logistic Regression']:
            if model_name in summary:
                model_summary = summary[model_name]
                assert 'test_accuracy' in model_summary
                assert 'cv_mean' in model_summary

    def test_invalid_input_handling(self, model):
        """Test handling of invalid inputs."""
        # Test prediction without training
        with pytest.raises(ValueError, match="Model has not been trained"):
            model.predict_single(np.array([1, 2, 3]))

        # Test training with empty data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):  # Should raise some exception
            model.train(empty_df, [])

    def test_evaluate_on_test_set(self, model, sample_data):
        """Test evaluation on separate test set."""
        features_df, labels = sample_data

        # Split data manually
        split_idx = len(features_df) // 2
        train_features = features_df.iloc[:split_idx]
        train_labels = labels[:split_idx]
        test_features = features_df.iloc[split_idx:]
        test_labels = labels[split_idx:]

        # Train model
        model.train(train_features, train_labels, models_to_train=['Random Forest'])

        # Evaluate on test set
        eval_results = model.evaluate_on_test_set(test_features, test_labels)

        assert 'accuracy' in eval_results
        assert 'precision' in eval_results
        assert 'recall' in eval_results
        assert 'f1_score' in eval_results
        assert 'confusion_matrix' in eval_results

        assert 0 <= eval_results['accuracy'] <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])