"""
Visualization Module

Creates comprehensive visualizations for wearable sensor data analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ResultVisualizer:
    """
    Comprehensive visualization class for wearable sensor data analysis results.

    Creates publication-ready plots and interactive visualizations.
    """

    def __init__(self, results_dir: str = "results", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize ResultVisualizer.

        Args:
            results_dir: Directory to save plots
            figsize: Default figure size
        """
        self.results_dir = results_dir
        self.figsize = figsize

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#1B998B'
        }

        logger.info(f"ResultVisualizer initialized. Results will be saved to {results_dir}")

    def plot_model_comparison(self, model_results: Dict[str, Dict],
                            save_name: str = "model_comparison.png") -> plt.Figure:
        """
        Create model performance comparison plots.

        Args:
            model_results: Dictionary with model results
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info("Creating model comparison plot...")

        # Prepare data
        models = list(model_results.keys())
        test_accuracies = [model_results[m].get('test_accuracy', model_results[m].get('accuracy', 0)) for m in models]
        cv_means = [model_results[m].get('cv_mean', 0) for m in models]
        cv_stds = [model_results[m].get('cv_std', 0) for m in models]
        f1_scores = [model_results[m].get('f1_score', 0) for m in models]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, test_accuracies, color=self.colors['primary'], alpha=0.7)
        ax1.errorbar(models, cv_means, yerr=cv_stds, fmt='o', color=self.colors['secondary'],
                    capsize=5, label='CV Mean ± Std')
        ax1.set_title('Test Accuracy vs Cross-Validation')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars1, test_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        # 2. Multiple metrics comparison
        ax2 = axes[0, 1]
        metrics_data = {
            'Accuracy': test_accuracies,
            'F1-Score': f1_scores,
            'CV Mean': cv_means
        }

        x = np.arange(len(models))
        width = 0.25

        for i, (metric, values) in enumerate(metrics_data.items()):
            ax2.bar(x + i*width, values, width, label=metric, alpha=0.8)

        ax2.set_title('Multiple Metrics Comparison')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('Models')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Cross-validation stability
        ax3 = axes[1, 0]
        ax3.errorbar(models, cv_means, yerr=cv_stds, fmt='o-', capsize=5,
                    color=self.colors['accent'], linewidth=2, markersize=8)
        ax3.set_title('Cross-Validation Stability')
        ax3.set_ylabel('CV Score ± Standard Deviation')
        ax3.set_xlabel('Models')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 4. Performance ranking
        ax4 = axes[1, 1]

        # Calculate composite score (weighted average)
        composite_scores = []
        for m in models:
            score = (0.4 * model_results[m].get('test_accuracy', 0) +
                    0.3 * model_results[m].get('f1_score', 0) +
                    0.3 * model_results[m].get('cv_mean', 0))
            composite_scores.append(score)

        # Sort by composite score
        sorted_indices = np.argsort(composite_scores)[::-1]
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [composite_scores[i] for i in sorted_indices]

        bars4 = ax4.barh(range(len(sorted_models)), sorted_scores,
                        color=plt.cm.viridis(np.linspace(0, 1, len(sorted_models))))
        ax4.set_title('Overall Performance Ranking')
        ax4.set_xlabel('Composite Score')
        ax4.set_yticks(range(len(sorted_models)))
        ax4.set_yticklabels(sorted_models)
        ax4.grid(True, alpha=0.3, axis='x')

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars4, sorted_scores)):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center')

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            title: str = "Confusion Matrix",
                            save_name: str = "confusion_matrix.png") -> plt.Figure:
        """
        Create an enhanced confusion matrix plot.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info("Creating confusion matrix plot...")

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'{title} - Absolute Counts')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. Normalized (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title(f'{title} - Normalized')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")

        return fig

    def plot_feature_importance(self, importance: np.ndarray, feature_names: List[str],
                              top_n: int = 20, title: str = "Feature Importance",
                              save_name: str = "feature_importance.png") -> plt.Figure:
        """
        Create feature importance visualization.

        Args:
            importance: Feature importance values
            feature_names: List of feature names
            top_n: Number of top features to show
            title: Plot title
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info(f"Creating feature importance plot (top {top_n} features)...")

        # Get top features
        top_indices = np.argsort(importance)[-top_n:]
        top_importance = importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 1. Horizontal bar chart
        bars = ax1.barh(range(len(top_names)), top_importance,
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_names))))
        ax1.set_title(f'{title} - Top {top_n} Features')
        ax1.set_xlabel('Importance')
        ax1.set_yticks(range(len(top_names)))
        ax1.set_yticklabels(top_names)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, top_importance)):
            ax1.text(bar.get_width() + max(top_importance)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{imp:.4f}', ha='left', va='center')

        # 2. Feature groups analysis (if feature names follow naming convention)
        feature_groups = self._group_features_by_type(top_names)

        if len(feature_groups) > 1:
            group_importance = {}
            for group, features in feature_groups.items():
                group_indices = [i for i, name in enumerate(top_names) if name in features]
                group_importance[group] = np.sum([top_importance[i] for i in group_indices])

            # Pie chart of feature group importance
            ax2.pie(group_importance.values(), labels=group_importance.keys(),
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Feature Group Contribution')
        else:
            # If no clear groups, show cumulative importance
            cumulative_importance = np.cumsum(top_importance[::-1]) / np.sum(top_importance)
            ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
                    'o-', color=self.colors['primary'])
            ax2.set_title('Cumulative Feature Importance')
            ax2.set_xlabel('Number of Top Features')
            ax2.set_ylabel('Cumulative Importance Ratio')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
            ax2.legend()

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def _group_features_by_type(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Group features by their type based on naming convention.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature types to feature names
        """
        groups = {
            'Statistical': [],
            'Time Domain': [],
            'Frequency Domain': [],
            'Peak Detection': [],
            'Cross-Axis': [],
            'Energy': []
        }

        for name in feature_names:
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['mean', 'std', 'var', 'min', 'max', 'median', 'q25', 'q75']):
                groups['Statistical'].append(name)
            elif any(keyword in name_lower for keyword in ['skew', 'kurt', 'rms', 'zero']):
                groups['Time Domain'].append(name)
            elif any(keyword in name_lower for keyword in ['freq', 'spectral', 'band', 'fft']):
                groups['Frequency Domain'].append(name)
            elif any(keyword in name_lower for keyword in ['peak', 'num_peaks']):
                groups['Peak Detection'].append(name)
            elif any(keyword in name_lower for keyword in ['correlation', 'svm', 'tilt']):
                groups['Cross-Axis'].append(name)
            elif any(keyword in name_lower for keyword in ['energy', 'power']):
                groups['Energy'].append(name)

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        return groups

    def plot_classification_report(self, classification_report: Dict,
                                 save_name: str = "classification_report.png") -> plt.Figure:
        """
        Visualize classification report metrics.

        Args:
            classification_report: Classification report dictionary
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info("Creating classification report visualization...")

        # Extract per-class metrics
        class_metrics = {}
        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                class_metrics[class_name] = metrics

        if not class_metrics:
            logger.warning("No valid class metrics found in classification report")
            return plt.figure()

        # Prepare data
        classes = list(class_metrics.keys())
        precision_scores = [class_metrics[c]['precision'] for c in classes]
        recall_scores = [class_metrics[c]['recall'] for c in classes]
        f1_scores = [class_metrics[c]['f1-score'] for c in classes]
        support_values = [class_metrics[c]['support'] for c in classes]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Classification Report Analysis', fontsize=16, fontweight='bold')

        # 1. Per-class metrics comparison
        ax1 = axes[0, 0]
        x = np.arange(len(classes))
        width = 0.25

        ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color=self.colors['primary'])
        ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8, color=self.colors['secondary'])
        ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color=self.colors['accent'])

        ax1.set_title('Per-Class Metrics')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Classes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. Support (sample count) visualization
        ax2 = axes[0, 1]
        bars = ax2.bar(classes, support_values, color=self.colors['info'], alpha=0.7)
        ax2.set_title('Class Support (Sample Count)')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xlabel('Classes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, support in zip(bars, support_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(support_values)*0.01,
                    f'{support}', ha='center', va='bottom')

        # 3. Precision vs Recall scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(recall_scores, precision_scores,
                            s=[s/10 for s in support_values], alpha=0.7,
                            c=f1_scores, cmap='viridis')

        # Add class labels
        for i, class_name in enumerate(classes):
            ax3.annotate(class_name, (recall_scores[i], precision_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax3.set_title('Precision vs Recall\n(Size: Support, Color: F1-Score)')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # Add diagonal line (perfect balance)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Balance')
        ax3.legend()

        # Colorbar for F1-scores
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('F1-Score')

        # 4. Radar chart for overall performance
        ax4 = axes[1, 1]

        # Calculate average metrics
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        # Add overall metrics from classification report if available
        macro_avg = classification_report.get('macro avg', {})
        weighted_avg = classification_report.get('weighted avg', {})

        if macro_avg and weighted_avg:
            categories = ['Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)',
                         'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1\n(Weighted)']
            values = [macro_avg.get('precision', 0), macro_avg.get('recall', 0), macro_avg.get('f1-score', 0),
                     weighted_avg.get('precision', 0), weighted_avg.get('recall', 0), weighted_avg.get('f1-score', 0)]
        else:
            categories = ['Avg Precision', 'Avg Recall', 'Avg F1-Score']
            values = [avg_precision, avg_recall, avg_f1]

        # Simple bar chart instead of radar for matplotlib compatibility
        bars = ax4.bar(range(len(categories)), values, color=self.colors['success'], alpha=0.7)
        ax4.set_title('Overall Performance Metrics')
        ax4.set_ylabel('Score')
        ax4.set_xticks(range(len(categories)))
        ax4.set_xticklabels(categories, rotation=45, ha='right')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Classification report plot saved to {save_path}")

        return fig

    def plot_data_distribution(self, df: pd.DataFrame, activity_col: str = 'activity',
                             save_name: str = "data_distribution.png") -> plt.Figure:
        """
        Visualize data distribution and basic statistics.

        Args:
            df: DataFrame with sensor data
            activity_col: Name of activity column
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info("Creating data distribution plots...")

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Activity distribution
        ax1 = fig.add_subplot(gs[0, :2])
        if activity_col in df.columns:
            activity_counts = df[activity_col].value_counts()
            bars = ax1.bar(activity_counts.index, activity_counts.values,
                          color=plt.cm.Set3(np.linspace(0, 1, len(activity_counts))))
            ax1.set_title('Activity Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)

            # Add percentage labels
            total_samples = activity_counts.sum()
            for bar, count in zip(bars, activity_counts.values):
                percentage = count / total_samples * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_samples*0.01,
                        f'{percentage:.1f}%', ha='center', va='bottom')

        # 2. Activity distribution pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        if activity_col in df.columns:
            ax2.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Activity Proportion')

        # 3-5. Accelerometer data distributions
        sensor_cols = ['acc_x', 'acc_y', 'acc_z']
        for i, col in enumerate(sensor_cols):
            if col in df.columns:
                ax = fig.add_subplot(gs[1, i])
                ax.hist(df[col], bins=50, alpha=0.7, color=self.colors['primary'])
                ax.set_title(f'{col.upper()} Distribution')
                ax.set_xlabel(f'{col} (g)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = df[col].mean()
                std_val = df[col].std()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ')
                ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
                ax.legend()

        # 6. Correlation heatmap
        ax6 = fig.add_subplot(gs[2, :2])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax6, cbar_kws={'shrink': 0.8})
            ax6.set_title('Feature Correlation Matrix')

        # 7. Basic statistics table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        if len(numeric_cols) > 0:
            stats_data = df[numeric_cols].describe().round(3)
            # Create table
            table_data = []
            for col in sensor_cols:
                if col in stats_data.columns:
                    table_data.append([
                        col.upper(),
                        f"{stats_data.loc['mean', col]:.3f}",
                        f"{stats_data.loc['std', col]:.3f}",
                        f"{stats_data.loc['min', col]:.3f}",
                        f"{stats_data.loc['max', col]:.3f}"
                    ])

            if table_data:
                table = ax7.table(cellText=table_data,
                                colLabels=['Sensor', 'Mean', 'Std', 'Min', 'Max'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                ax7.set_title('Summary Statistics', pad=20)

        plt.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Data distribution plot saved to {save_path}")

        return fig

    def create_interactive_dashboard(self, model_results: Dict, features_df: pd.DataFrame,
                                   labels: List[str], save_name: str = "interactive_dashboard.html"):
        """
        Create an interactive dashboard using Plotly.

        Args:
            model_results: Dictionary with model results
            features_df: Features DataFrame
            labels: Activity labels
            save_name: Name of the saved HTML file
        """
        logger.info("Creating interactive dashboard...")

        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance', 'Feature Importance',
                              'Class Distribution', 'Feature Correlation'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'pie'}, {'type': 'heatmap'}]]
            )

            # 1. Model performance comparison
            models = list(model_results.keys())
            accuracies = [model_results[m].get('test_accuracy', model_results[m].get('accuracy', 0)) for m in models]

            fig.add_trace(
                go.Bar(x=models, y=accuracies, name='Accuracy',
                      text=[f'{acc:.3f}' for acc in accuracies],
                      textposition='auto'),
                row=1, col=1
            )

            # 2. Feature importance (top 10)
            best_model = max(model_results.keys(), key=lambda k: model_results[k].get('test_accuracy', 0))
            if 'feature_importance' in model_results[best_model] and model_results[best_model]['feature_importance'] is not None:
                importance = model_results[best_model]['feature_importance']
                feature_names = features_df.columns.tolist()

                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = importance[top_indices]

                fig.add_trace(
                    go.Bar(y=top_features, x=top_importance, orientation='h',
                          name='Importance', text=[f'{imp:.4f}' for imp in top_importance],
                          textposition='auto'),
                    row=1, col=2
                )

            # 3. Class distribution
            label_counts = pd.Series(labels).value_counts()
            fig.add_trace(
                go.Pie(labels=label_counts.index, values=label_counts.values,
                      name="Class Distribution"),
                row=2, col=1
            )

            # 4. Feature correlation heatmap (subset)
            if len(features_df.columns) > 20:
                # Select top 20 features for correlation
                corr_features = features_df.columns[:20]
            else:
                corr_features = features_df.columns

            corr_matrix = features_df[corr_features].corr()

            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_features,
                          y=corr_features,
                          colorscale='RdBu',
                          zmid=0),
                row=2, col=2
            )

            # Update layout
            fig.update_layout(
                title_text="Wearable Sensor Data Analysis Dashboard",
                title_x=0.5,
                height=800,
                showlegend=False
            )

            # Save interactive plot
            save_path = os.path.join(self.results_dir, save_name)
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {str(e)}")
            logger.info("Falling back to static plots...")

    def plot_learning_curves(self, model_results: Dict, save_name: str = "learning_curves.png") -> plt.Figure:
        """
        Plot learning curves if cross-validation data is available.

        Args:
            model_results: Dictionary with model results
            save_name: Name of the saved plot file

        Returns:
            Matplotlib figure
        """
        logger.info("Creating learning curves plot...")

        fig, axes = plt.subplots(1, len(model_results), figsize=(5*len(model_results), 5))
        if len(model_results) == 1:
            axes = [axes]

        for i, (model_name, results) in enumerate(model_results.items()):
            ax = axes[i]

            # Plot training and validation scores
            train_acc = results.get('train_accuracy', 0)
            test_acc = results.get('test_accuracy', results.get('accuracy', 0))
            cv_scores = results.get('cv_scores', [])

            if cv_scores:
                # Plot CV scores distribution
                ax.boxplot([cv_scores], labels=['CV Scores'])
                ax.scatter([1], [test_acc], color='red', s=100, label=f'Test: {test_acc:.3f}', zorder=5)
                ax.scatter([1], [train_acc], color='blue', s=100, label=f'Train: {train_acc:.3f}', zorder=5)
            else:
                # Simple bar chart
                ax.bar(['Train', 'Test'], [train_acc, test_acc],
                      color=[self.colors['primary'], self.colors['secondary']], alpha=0.7)

            ax.set_title(f'{model_name}\nLearning Performance')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves plot saved to {save_path}")

        return fig

    def generate_complete_report(self, model_results: Dict, features_df: pd.DataFrame,
                               labels: List[str], raw_data: Optional[pd.DataFrame] = None):
        """
        Generate a complete visual report with all plots.

        Args:
            model_results: Dictionary with model results
            features_df: Features DataFrame
            labels: Activity labels
            raw_data: Optional raw sensor data
        """
        logger.info("Generating complete visual report...")

        # 1. Model comparison
        self.plot_model_comparison(model_results, "01_model_comparison.png")

        # 2. Best model confusion matrix
        best_model = max(model_results.keys(), key=lambda k: model_results[k].get('test_accuracy', 0))
        best_results = model_results[best_model]

        if 'confusion_matrix' in best_results:
            activity_names = list(set(labels))
            self.plot_confusion_matrix(best_results['confusion_matrix'], activity_names,
                                     f"Confusion Matrix - {best_model}", "02_confusion_matrix.png")

        # 3. Feature importance
        if 'feature_importance' in best_results and best_results['feature_importance'] is not None:
            self.plot_feature_importance(best_results['feature_importance'], features_df.columns.tolist(),
                                       title=f"Feature Importance - {best_model}", save_name="03_feature_importance.png")

        # 4. Classification report
        if 'classification_report' in best_results:
            self.plot_classification_report(best_results['classification_report'], "04_classification_report.png")

        # 5. Data distribution (if raw data provided)
        if raw_data is not None:
            self.plot_data_distribution(raw_data, save_name="05_data_distribution.png")

        # 6. Learning curves
        self.plot_learning_curves(model_results, "06_learning_curves.png")

        # 7. Interactive dashboard
        self.create_interactive_dashboard(model_results, features_df, labels, "07_interactive_dashboard.html")

        logger.info("Complete visual report generated successfully!")
        logger.info(f"All plots saved to {self.results_dir} directory")