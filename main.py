#!/usr/bin/env python3
"""
Main script for Wearable Data Analysis Pipeline

This script orchestrates the complete machine learning pipeline for 
wearable sensor data analysis and activity classification.
"""

import os
import sys
import argparse
import logging
from typing import Optional, List
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import WearableModel
from src.evaluator import ModelEvaluator
from src.visualizer import ResultVisualizer
from src.load_uci_har import UCIHARLoader, load_uci_har_for_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wearable_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wearable Sensor Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generate-data --n-samples 15000
  python main.py --data-file data/processed/sensor_data.csv
  python main.py --use-uci-har
  python main.py --use-uci-har --uci-har-raw-signals
  python main.py --generate-data --models "Random Forest" "Logistic Regression"
  python main.py --data-file data.csv --window-size 50 --overlap 0.3
        """
    )

    # Data options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data-file', type=str,
                            help='Path to CSV file with sensor data')
    data_group.add_argument('--generate-data', action='store_true',
                            help='Generate synthetic sensor data')
    data_group.add_argument('--use-uci-har', action='store_true',
                            help='Use UCI HAR dataset (downloads automatically)')

    # UCI HAR specific options
    parser.add_argument('--uci-har-path', type=str, default='data/raw/UCI_HAR',
                        help='Path for UCI HAR dataset (default: data/raw/UCI_HAR)')
    parser.add_argument('--uci-har-raw-signals', action='store_true',
                        help='Include raw sensor signals from UCI HAR')

    # Data generation options
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--sampling-rate', type=float, default=25.0,
                        help='Sampling rate in Hz (default: 25.0)')

    # Feature engineering options
    parser.add_argument('--window-size', type=int, default=25,
                        help='Window size for feature extraction (default: 25)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Window overlap ratio (default: 0.5)')

    # Model options
    parser.add_argument('--models', nargs='+',
                        choices=['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM'],
                        help='Models to train (default: all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size ratio (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')

    # Output options
    parser.add_argument('--save-model', type=str,
                        help='Path to save the best trained model')
    parser.add_argument('--save-features', type=str,
                        help='Path to save extracted features')
    parser.add_argument('--save-report', type=str,
                        help='Path to save evaluation report')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    # Processing options
    parser.add_argument('--group-by', type=str,
                        help='Column to group by during feature extraction (e.g., user_id)')
    parser.add_argument('--activity-col', type=str, default='activity',
                        help='Name of activity column (default: activity)')

    return parser.parse_args()


def load_or_generate_data(args, data_loader: DataLoader) -> pd.DataFrame:
    """Load data from file, generate synthetic data, or use UCI HAR dataset."""
    if args.generate_data:
        logger.info("Generating synthetic sensor data...")
        raw_data = data_loader.generate_sample_data(
            n_samples=args.n_samples,
            sampling_rate=args.sampling_rate
        )
    elif args.use_uci_har:
        logger.info("Loading UCI HAR dataset...")
        try:
            raw_data = load_uci_har_for_pipeline(
                data_dir=args.uci_har_path,
                auto_download=True,
                convert_format=True,
                include_raw_signals=args.uci_har_raw_signals
            )
            logger.info(f"UCI HAR dataset loaded successfully: {raw_data.shape}")

            # Print UCI HAR dataset info
            loader = UCIHARLoader(args.uci_har_path)
            dataset_info = loader.get_dataset_info()
            logger.info(f"Dataset: {dataset_info['name']}")
            logger.info(f"Activities: {dataset_info['activities']}")
            logger.info(f"Subjects: {dataset_info['n_subjects']}")

        except Exception as e:
            logger.error(f"Failed to load UCI HAR dataset: {str(e)}")
            logger.info("Falling back to synthetic data generation...")
            raw_data = data_loader.generate_sample_data(
                n_samples=args.n_samples,
                sampling_rate=args.sampling_rate
            )
    else:
        logger.info(f"Loading data from {args.data_file}")
        raw_data = data_loader.load_csv_data(args.data_file)

        # Validate data format
        is_valid, issues = data_loader.validate_data_format(raw_data)
        if not is_valid:
            logger.error("Data validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            raise ValueError("Invalid data format")

    return raw_data


def run_pipeline(args):
    """Run the complete wearable data analysis pipeline."""
    logger.info("Starting Wearable Data Analysis Pipeline")
    logger.info("=" * 60)

    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer(
        window_size=args.window_size,
        overlap=args.overlap,
        sampling_rate=args.sampling_rate
    )
    model = WearableModel()
    evaluator = ModelEvaluator()
    visualizer = ResultVisualizer()

    try:
        # Step 1: Load/Generate Data
        logger.info("Step 1: Loading/Generating Data")
        raw_data = load_or_generate_data(args, data_loader)

        # Print data summary
        summary = data_loader.get_data_summary(raw_data)
        logger.info(f"Data shape: {summary['shape']}")
        if 'activity_distribution' in summary:
            logger.info(f"Activity distribution: {summary['activity_distribution']}")

        # Step 2: Preprocess Data
        logger.info("Step 2: Preprocessing Data")
        processed_data = data_loader.preprocess_data(raw_data)

        # Step 3: Feature Engineering
        logger.info("Step 3: Feature Engineering")
        features_df, labels = feature_engineer.extract_features(
            processed_data,
            group_by=args.group_by
        )

        if len(features_df) == 0:
            raise ValueError("No features extracted. Check your data and parameters.")

        # Save features if requested
        if args.save_features:
            feature_engineer.save_features(features_df, labels, args.save_features)
            logger.info(f"Features saved to {args.save_features}")

        # Step 4: Model Training
        logger.info("Step 4: Training Models")
        model_results = model.train(
            features_df,
            labels,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            models_to_train=args.models
        )

        if not model_results:
            raise ValueError("No models were successfully trained")

        # Step 5: Model Evaluation
        logger.info("Step 5: Evaluating Models")

        # Set class names for evaluator
        unique_activities = list(set(labels))
        evaluator.class_names = unique_activities

        # Evaluate each model
        evaluation_results = {}
        for model_name, result in model_results.items():
            eval_result = evaluator.evaluate_model(
                result['y_test'],
                result['y_test_pred'],
                result.get('y_test_proba'),
                model_name
            )
            evaluation_results[model_name] = eval_result

        # Step 6: Generate Report
        logger.info("Step 6: Generating Performance Report")

        # Generate text report
        report_text = evaluator.generate_performance_report(model_results)
        print("\n" + report_text)

        # Save report if requested
        if args.save_report:
            with open(args.save_report, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {args.save_report}")

        # Step 7: Visualizations
        if not args.no_plots:
            logger.info("Step 7: Generating Visualizations")
            visualizer.generate_complete_report(
                model_results,
                features_df,
                labels,
                raw_data
            )

        # Step 8: Save Model
        if args.save_model:
            logger.info("Step 8: Saving Best Model")
            model.save_model(args.save_model)
            logger.info(f"Best model saved to {args.save_model}")

        # Final Summary
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)

        best_model_name = model.best_model_name
        best_accuracy = model_results[best_model_name]['test_accuracy']
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best Accuracy: {best_accuracy:.4f}")
        logger.info(f"Total Features: {len(features_df.columns)}")
        logger.info(f"Total Samples: {len(features_df)}")
        logger.info(f"Activities: {', '.join(unique_activities)}")

        # Feature importance summary
        feature_importance_df = model.get_feature_importance(top_n=10)
        if not feature_importance_df.empty:
            logger.info("\nTop 10 Most Important Features:")
            for _, row in feature_importance_df.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'model': model,
            'results': model_results,
            'features': features_df,
            'labels': labels,
            'raw_data': raw_data,
            'evaluator': evaluator,
            'visualizer': visualizer
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def predict_activity_example(pipeline_results):
    """Example of how to use the trained model for prediction."""
    logger.info("\nExample: Making predictions on new data")

    model = pipeline_results['model']
    features_df = pipeline_results['features']

    # Use a sample from the feature set as an example
    sample_features = features_df.iloc[0].values

    try:
        activity, confidence = model.predict_single(sample_features)
        logger.info(f"Predicted Activity: {activity}")
        logger.info(f"Confidence: {confidence:.4f}")

        # Batch prediction example
        sample_batch = features_df.iloc[:5]
        activities, probabilities = model.predict(sample_batch)

        logger.info("\nBatch prediction example:")
        for i, (activity, proba) in enumerate(zip(activities, probabilities)):
            max_conf = max(proba)
            logger.info(f"  Sample {i + 1}: {activity} (confidence: {max_conf:.4f})")

    except Exception as e:
        logger.error(f"Prediction example failed: {str(e)}")


def main():
    """Main function."""
    args = parse_arguments()

    try:
        # Run the pipeline
        pipeline_results = run_pipeline(args)

        # Show prediction example
        predict_activity_example(pipeline_results)

        # Additional analysis suggestions
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS & SUGGESTIONS:")
        logger.info("=" * 60)
        logger.info("1. Check the results/ directory for detailed visualizations")
        logger.info("2. Review the classification report for per-activity performance")
        logger.info("3. Consider feature selection if you have many features")
        logger.info("4. Try different window sizes and overlap ratios")
        logger.info("5. Collect more data for underperforming activities")
        logger.info("6. Experiment with ensemble methods for better performance")

        if args.save_model:
            logger.info(f"7. Load the saved model using: model.load_model('{args.save_model}')")

        logger.info("\nFor real-time deployment, consider:")
        logger.info("- Implementing sliding window processing")
        logger.info("- Adding data validation and error handling")
        logger.info("- Setting up model monitoring and retraining")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()