"""
Machine Learning Model Module

Handles training, evaluation, and prediction for wearable sensor data classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WearableModel:
    """
    Machine learning model class for wearable sensor data classification.

    Supports multiple algorithms with hyperparameter tuning and proper validation.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize WearableModel.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.is_trained = False

        # Define model configurations
        self.model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2'],
                    'model__solver': ['liblinear', 'saga']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 'log2']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.05, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=random_state, probability=True),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['rbf', 'linear'],
                    'model__gamma': ['scale', 'auto']
                }
            }
        }

        logger.info(f"WearableModel initialized with {len(self.model_configs)} algorithms")

    def train(self, X: pd.DataFrame, y: List[str], test_size: float = 0.2,
              cv_folds: int = 5, models_to_train: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train multiple machine learning models with hyperparameter tuning.

        Args:
            X: Feature DataFrame
            y: List of activity labels
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            models_to_train: List of model names to train (None for all)

        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

        # Determine which models to train
        if models_to_train is None:
            models_to_train = list(self.model_configs.keys())

        results = {}

        # Train each model
        for model_name in models_to_train:
            if model_name not in self.model_configs:
                logger.warning(f"Unknown model: {model_name}")
                continue

            logger.info(f"Training {model_name}...")

            try:
                result = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test, cv_folds
                )
                results[model_name] = result
                logger.info(f"{model_name} - Accuracy: {result['test_accuracy']:.4f}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue

        # Select best model
        if results:
            self.best_model_name = max(results.keys(),
                                       key=lambda k: results[k]['test_accuracy'])
            self.best_model = results[self.best_model_name]['pipeline']
            self.is_trained = True

            logger.info(f"Best model: {self.best_model_name} "
                        f"(Accuracy: {results[self.best_model_name]['test_accuracy']:.4f})")

        self.models = results
        return results

    def _train_single_model(self, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: np.ndarray, y_test: np.ndarray, cv_folds: int) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.

        Args:
            model_name: Name of the model to train
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with model results
        """
        config = self.model_configs[model_name]

        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline,
            config['params'],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get best pipeline
        best_pipeline = grid_search.best_estimator_

        # Make predictions
        y_train_pred = best_pipeline.predict(X_train)
        y_test_pred = best_pipeline.predict(X_test)

        # Get prediction probabilities for multiclass
        try:
            y_test_proba = best_pipeline.predict_proba(X_test)
        except:
            y_test_proba = None

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Cross-validation scores
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring='accuracy')

        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted'
        )

        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_test_pred, target_names=class_names, output_dict=True
        )

        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_pipeline.named_steps['model'], 'feature_importances_'):
            feature_importance = best_pipeline.named_steps['model'].feature_importances_
        elif hasattr(best_pipeline.named_steps['model'], 'coef_'):
            # For linear models, use absolute coefficients
            coef = best_pipeline.named_steps['model'].coef_
            if len(coef.shape) > 1:
                # Multi-class: average across classes
                feature_importance = np.mean(np.abs(coef), axis=0)
            else:
                feature_importance = np.abs(coef)

        return {
            'pipeline': best_pipeline,
            'best_params': grid_search.best_params_,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }

    def predict(self, X: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predicted_activities, prediction_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Make predictions
        y_pred_encoded = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)

        # Convert back to activity names
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        return y_pred.tolist(), y_pred_proba

    def predict_single(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict activity for a single sample.

        Args:
            features: 1D array of features

        Returns:
            Tuple of (predicted_activity, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Reshape for prediction
        features_reshaped = features.reshape(1, -1)

        # Make prediction
        y_pred_encoded = self.best_model.predict(features_reshaped)[0]
        y_pred_proba = self.best_model.predict_proba(features_reshaped)[0]

        # Convert to activity name
        activity = self.label_encoder.inverse_transform([y_pred_encoded])[0]
        confidence = np.max(y_pred_proba)

        return activity, confidence

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models.

        Returns:
            Dictionary with model performance summary
        """
        if not self.models:
            return {}

        summary = {}

        for name, result in self.models.items():
            summary[name] = {
                'test_accuracy': result['test_accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'best_params': result['best_params']
            }

        summary['best_model'] = self.best_model_name
        return summary

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the best model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        best_result = self.models[self.best_model_name]
        feature_importance = best_result['feature_importance']

        if feature_importance is None:
            logger.warning(f"Feature importance not available for {self.best_model_name}")
            return pd.DataFrame()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'models_summary': self.get_model_summary()
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model.

        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Best model: {self.best_model_name}")

    def cross_validate_model(self, X: pd.DataFrame, y: List[str],
                             model_name: str, cv_folds: int = 10) -> Dict[str, float]:
        """
        Perform detailed cross-validation for a specific model.

        Args:
            X: Feature DataFrame
            y: List of activity labels
            model_name: Name of the model to validate
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation results
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Create pipeline
        config = self.model_configs[model_name]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', config['model'])
        ])

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        # Perform cross-validation with multiple metrics
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        results = {}
        for score in scoring:
            scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring=score)
            results[f'{score}_mean'] = scores.mean()
            results[f'{score}_std'] = scores.std()
            results[f'{score}_scores'] = scores

        logger.info(f"Cross-validation completed for {model_name}")
        return results

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of classes in the training data.

        Returns:
            Dictionary mapping class names to counts
        """
        if not hasattr(self.label_encoder, 'classes_'):
            return {}

        return dict(zip(self.label_encoder.classes_,
                        range(len(self.label_encoder.classes_))))

    def evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: List[str]) -> Dict[str, Any]:
        """
        Evaluate the best model on a separate test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")

        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)

        # Make predictions
        y_pred_encoded = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, y_pred_encoded, average='weighted'
        )

        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test_encoded, y_pred_encoded, target_names=class_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_true': y_test_encoded,
            'y_pred': y_pred_encoded,
            'y_pred_proba': y_pred_proba
        }

        logger.info(f"Test set evaluation - Accuracy: {accuracy:.4f}")
        return results