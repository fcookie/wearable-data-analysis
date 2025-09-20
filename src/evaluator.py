"""
Model Evaluation Module

Provides comprehensive evaluation metrics and analysis for wearable sensor classification models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           confusion_matrix, classification_report, roc_auc_score,
                           roc_curve, precision_recall_curve, average_precision_score)
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for wearable sensor data classification.

    Provides detailed metrics, statistical analysis, and performance comparisons.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize ModelEvaluator.

        Args:
            class_names: List of class/activity names
        """
        self.class_names = class_names
        self.evaluation_results = {}

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Classification report
        if self.class_names:
            target_names = self.class_names
        else:
            target_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

        class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

        # Per-class metrics
        per_class_metrics = {}
        unique_classes = np.unique(y_true)

        for i, class_idx in enumerate(unique_classes):
            class_name = target_names[i] if i < len(target_names) else f"Class_{class_idx}"
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i, class_idx in enumerate(unique_classes):
            class_mask = (y_true == class_idx)
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                class_name = target_names[i] if i < len(target_names) else f"Class_{class_idx}"
                per_class_accuracy[class_name] = class_accuracy

        # Additional metrics if probabilities are available
        auc_scores = {}
        if y_pred_proba is not None:
            try:
                # Multi-class AUC (one-vs-rest)
                if len(unique_classes) > 2:
                    auc_scores['macro'] = roc_auc_score(y_true, y_pred_proba,
                                                       multi_class='ovr', average='macro')
                    auc_scores['weighted'] = roc_auc_score(y_true, y_pred_proba,
                                                          multi_class='ovr', average='weighted')
                else:
                    # Binary classification
                    auc_scores['binary'] = roc_auc_score(y_true, y_pred_proba[:, 1])

                # Per-class AUC
                for i, class_idx in enumerate(unique_classes):
                    class_name = target_names[i] if i < len(target_names) else f"Class_{class_idx}"
                    y_true_binary = (y_true == class_idx).astype(int)
                    if len(np.unique(y_true_binary)) > 1:  # Check if class exists in true labels
                        auc_scores[f'{class_name}_vs_rest'] = roc_auc_score(
                            y_true_binary, y_pred_proba[:, i]
                        )
            except Exception as e:
                logger.warning(f"Could not calculate AUC scores: {str(e)}")

        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': per_class_metrics,
            'per_class_accuracy': per_class_accuracy,
            'auc_scores': auc_scores,
            'support_total': np.sum(support),
            'n_classes': len(unique_classes)
        }

        # Store results
        self.evaluation_results[model_name] = results

        logger.info(f"{model_name} evaluation completed - Accuracy: {accuracy:.4f}")
        return results

    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models and rank them by performance.

        Args:
            model_results: Dictionary mapping model names to their results

        Returns:
            DataFrame with model comparison
        """
        logger.info(f"Comparing {len(model_results)} models...")

        comparison_data = []

        for model_name, results in model_results.items():
            comparison_data.append({
                'model': model_name,
                'accuracy': results.get('test_accuracy', results.get('accuracy', 0)),
                'precision_weighted': results.get('precision', 0),
                'recall_weighted': results.get('recall', 0),
                'f1_weighted': results.get('f1_score', 0),
                'cv_mean': results.get('cv_mean', 0),
                'cv_std': results.get('cv_std', 0)
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)

        # Add ranking
        comparison_df['rank'] = range(1, len(comparison_df) + 1)

        return comparison_df

    def calculate_statistical_significance(self, results1: Dict, results2: Dict,
                                         metric: str = 'accuracy') -> Dict[str, float]:
        """
        Calculate statistical significance between two models.

        Args:
            results1: Results from first model
            results2: Results from second model
            metric: Metric to compare

        Returns:
            Dictionary with statistical test results
        """
        from scipy.stats import ttest_rel, wilcoxon

        # Extract CV scores if available
        scores1 = results1.get('cv_scores', [results1.get(metric, 0)])
        scores2 = results2.get('cv_scores', [results2.get(metric, 0)])

        if len(scores1) != len(scores2) or len(scores1) < 3:
            logger.warning("Insufficient data for statistical significance testing")
            return {}

        # Paired t-test
        t_stat, t_pvalue = ttest_rel(scores1, scores2)

        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = wilcoxon(scores1, scores2)
        except:
            w_stat, w_pvalue = np.nan, np.nan

        return {
            'metric': metric,
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'significant_at_0.05': t_pvalue < 0.05
        }

    def analyze_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detailed analysis of confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names

        Returns:
            Dictionary with confusion matrix analysis
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]

        n_classes = cm.shape[0]
        total_samples = np.sum(cm)

        # Overall accuracy
        accuracy = np.trace(cm) / total_samples

        # Per-class analysis
        class_analysis = {}
        for i in range(n_classes):
            class_name = class_names[i]

            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total_samples - tp - fp - fn

            # Metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_analysis[class_name] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'support': tp + fn
            }

        # Find most confused classes
        confused_pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': cm[i, j],
                        'percentage': cm[i, j] / np.sum(cm[i, :]) * 100
                    })

        # Sort by count
        confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)

        return {
            'overall_accuracy': accuracy,
            'total_samples': total_samples,
            'class_analysis': class_analysis,
            'most_confused_pairs': confused_pairs[:10],  # Top 10
            'confusion_matrix': cm.tolist()
        }

    def generate_performance_report(self, model_results: Dict[str, Dict],
                                  save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.

        Args:
            model_results: Dictionary with model results
            save_path: Optional path to save the report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("WEARABLE SENSOR DATA CLASSIFICATION - PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Model comparison
        comparison_df = self.compare_models(model_results)
        report_lines.append("MODEL COMPARISON SUMMARY:")
        report_lines.append("-" * 40)

        for _, row in comparison_df.iterrows():
            report_lines.append(f"{row['rank']}. {row['model']}")
            report_lines.append(f"   Accuracy: {row['accuracy']:.4f}")
            report_lines.append(f"   F1-Score: {row['f1_weighted']:.4f}")
            report_lines.append(f"   CV Score: {row['cv_mean']:.4f} ± {row['cv_std']:.4f}")
            report_lines.append("")

        # Detailed analysis for best model
        best_model = comparison_df.iloc[0]['model']
        best_results = model_results[best_model]

        report_lines.append(f"DETAILED ANALYSIS - BEST MODEL ({best_model}):")
        report_lines.append("-" * 40)

        if 'classification_report' in best_results:
            report_lines.append("Classification Report:")
            class_report = best_results['classification_report']

            # Format classification report
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    report_lines.append(f"  {class_name}:")
                    report_lines.append(f"    Precision: {metrics['precision']:.3f}")
                    report_lines.append(f"    Recall: {metrics['recall']:.3f}")
                    report_lines.append(f"    F1-Score: {metrics['f1-score']:.3f}")
                    report_lines.append(f"    Support: {metrics['support']}")
                    report_lines.append("")

        # Confusion matrix analysis
        if 'confusion_matrix' in best_results:
            cm = best_results['confusion_matrix']
            cm_analysis = self.analyze_confusion_matrix(cm, self.class_names)

            report_lines.append("Confusion Matrix Analysis:")
            report_lines.append(f"  Total Samples: {cm_analysis['total_samples']}")
            report_lines.append(f"  Overall Accuracy: {cm_analysis['overall_accuracy']:.4f}")
            report_lines.append("")

            if cm_analysis['most_confused_pairs']:
                report_lines.append("Most Confused Class Pairs:")
                for pair in cm_analysis['most_confused_pairs'][:5]:
                    report_lines.append(f"  {pair['true_class']} → {pair['predicted_class']}: "
                                      f"{pair['count']} samples ({pair['percentage']:.1f}%)")
                report_lines.append("")

        # Feature importance (if available)
        if 'feature_importance' in best_results and best_results['feature_importance'] is not None:
            importance = best_results['feature_importance']
            if hasattr(self, 'feature_names') and self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f"Feature_{i}" for i in range(len(importance))]

            # Get top 10 features
            top_indices = np.argsort(importance)[-10:][::-1]

            report_lines.append("Top 10 Most Important Features:")
            for i, idx in enumerate(top_indices):
                feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                report_lines.append(f"  {i+1}. {feature_name}: {importance[idx]:.4f}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)

        best_accuracy = comparison_df.iloc[0]['accuracy']
        if best_accuracy < 0.7:
            report_lines.append("• Model performance is below 70%. Consider:")
            report_lines.append("  - Collecting more training data")
            report_lines.append("  - Feature engineering improvements")
            report_lines.append("  - Trying different algorithms")
        elif best_accuracy < 0.85:
            report_lines.append("• Model performance is moderate. Consider:")
            report_lines.append("  - Hyperparameter optimization")
            report_lines.append("  - Ensemble methods")
            report_lines.append("  - Data quality improvements")
        else:
            report_lines.append("• Model performance is good!")
            report_lines.append("  - Consider deployment readiness")
            report_lines.append("  - Monitor performance on new data")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report saved to {save_path}")

        return report_text

    def calculate_model_robustness(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                 confidence_threshold: float = 0.8) -> Dict[str, float]:
        """
        Calculate model robustness metrics.

        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            confidence_threshold: Threshold for high-confidence predictions

        Returns:
            Dictionary with robustness metrics
        """
        # Prediction confidence
        max_proba = np.max(y_pred_proba, axis=1)
        mean_confidence = np.mean(max_proba)

        # High-confidence predictions
        high_conf_mask = max_proba >= confidence_threshold
        high_conf_ratio = np.mean(high_conf_mask)

        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(
                y_true[high_conf_mask],
                np.argmax(y_pred_proba[high_conf_mask], axis=1)
            )
        else:
            high_conf_accuracy = 0

        # Prediction entropy (uncertainty)
        epsilon = 1e-15  # Avoid log(0)
        entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + epsilon), axis=1)
        mean_entropy = np.mean(entropy)

        return {
            'mean_confidence': mean_confidence,
            'high_confidence_ratio': high_conf_ratio,
            'high_confidence_accuracy': high_conf_accuracy,
            'mean_prediction_entropy': mean_entropy,
            'confidence_threshold': confidence_threshold
        }

    def analyze_class_balance(self, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Analyze class balance in the dataset.

        Args:
            y_true: True labels

        Returns:
            Dictionary with class balance analysis
        """
        unique_classes, counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)

        class_distribution = {}
        for class_idx, count in zip(unique_classes, counts):
            class_name = self.class_names[class_idx] if (self.class_names and
                                                        class_idx < len(self.class_names)) else f"Class_{class_idx}"
            class_distribution[class_name] = {
                'count': count,
                'percentage': count / total_samples * 100
            }

        # Calculate imbalance metrics
        max_count = np.max(counts)
        min_count = np.min(counts)
        imbalance_ratio = max_count / min_count

        # Gini coefficient for inequality
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts) - (n + 1) * np.sum(sorted_counts)) / (n * np.sum(sorted_counts))

        return {
            'class_distribution': class_distribution,
            'total_samples': total_samples,
            'n_classes': len(unique_classes),
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini,
            'is_balanced': imbalance_ratio <= 3.0  # Rule of thumb
        }