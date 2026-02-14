"""
Time-Series AutoML Pipeline
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from timeseries.ts_detector import TimeSeriesDetector
from timeseries.ts_preprocessor import TimeSeriesPreprocessor
from timeseries.ts_models import TimeSeriesModelZoo
from timeseries.ts_evaluator import TimeSeriesEvaluator
from tuner.optuna_tuner import OptunaHyperparameterTuner
from utils.utils import create_artifacts_dir, save_json
from config.config import ARTIFACTS_DIR, TIMESERIES_MODELS, OPTUNA_CONFIG


def run_timeseries_automl(csv_path: str, target_column: str, date_column: str = None,
                          test_size: float = 0.2, sequence_length: int = 10):
    """
    Run time-series AutoML pipeline with comprehensive error handling
    
    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
        date_column: Optional date column name
        test_size: Fraction for test set
        sequence_length: Sequence length for deep learning models
        
    Returns:
        Dictionary with results
    """
    try:
        logger.info("="*80)
        logger.info("TIME-SERIES AUTOML PIPELINE STARTED")
        logger.info("="*80)
        print("="*80)
        print("TIME-SERIES AUTOML PIPELINE STARTED")
        print("="*80)
        
        # Create artifacts directory
        try:
            artifacts_path = create_artifacts_dir(ARTIFACTS_DIR)
        except Exception as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            raise RuntimeError(f"Could not create artifacts directory: {e}")
        
        # Step 1: Load data
        print("\n[1/7] Loading data...")
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise RuntimeError(f"Failed to load data: {e}")
        
        # Step 2: Detect time-series
        print("\n[2/7] Detecting time-series patterns...")
        detector = TimeSeriesDetector()
        is_ts = detector.detect(df, date_column=date_column, target_column=target_column)
        
        if not is_ts:
            raise ValueError("Dataset is not time-series. Use regular AutoML pipeline instead.")
        
        ts_info = detector.get_info()
        print(f"Time-series detected:")
        print(f"  Date column: {ts_info['date_column']}")
        print(f"  Frequency: {ts_info['frequency']}")
        print(f"  Has seasonality: {ts_info['has_seasonality']}")
        
        # Step 3: Prepare data
        print("\n[3/7] Preparing time-series data...")
        df_prepared = detector.prepare_for_modeling(df, target_column)
        
        preprocessor = TimeSeriesPreprocessor(sequence_length=sequence_length)
        train_df, test_df = preprocessor.train_test_split_temporal(df_prepared, test_size=test_size)
        
        print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
        
        # Step 4: Initialize models
        print("\n[4/7] Initializing time-series models...")
        model_zoo = TimeSeriesModelZoo()
        models = model_zoo.get_all_models()
        
        # Step 5: Train and evaluate models
        print("\n[5/7] Training and evaluating models...")
        predictions = {}
        
        # Get target series
        y_train = train_df[target_column]
        y_test = test_df[target_column]
        
        # Train each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                if model_name == 'arima':
                    # ARIMA
                    model.fit(y_train)
                    y_pred = model.predict(n_periods=len(y_test))
                    predictions[model_name] = y_pred
                
                elif model_name == 'prophet':
                    # Prophet
                    prophet_train = preprocessor.prepare_for_prophet(train_df, target_column)
                    model.fit(prophet_train)
                    forecast = model.predict(periods=len(y_test), freq=ts_info['frequency'] or 'D')
                    predictions[model_name] = forecast['yhat'].values[-len(y_test):]
                
                elif model_name in ['lstm', 'gru', 'bidirectional_lstm']:
                    # Deep learning models
                    X_train, y_train_seq = preprocessor.prepare_for_deep_learning(y_train, scale=True)
                    X_test, y_test_seq = preprocessor.prepare_for_deep_learning(y_test, scale=True)
                    
                    # Split for validation
                    val_split = int(len(X_train) * 0.8)
                    X_train_dl, X_val = X_train[:val_split], X_train[val_split:]
                    y_train_dl, y_val = y_train_seq[:val_split], y_train_seq[val_split:]
                    
                    model.fit(X_train_dl, y_train_dl, X_val, y_val)
                    y_pred_scaled = model.predict(X_test)
                    
                    # Inverse scale
                    y_pred = preprocessor.inverse_scale(y_pred_scaled.flatten())
                    predictions[model_name] = y_pred[:len(y_test_seq)]
                
                print(f"✓ {model_name} trained successfully")
            
            except Exception as e:
                msg = f"{model_name} failed: {e}"
                logger.warning(msg)
                print(f"✗ {msg}")
                continue
        
        # Step 6: Evaluate all models
        print("\n[6/7] Evaluating models...")
        evaluator = TimeSeriesEvaluator()
        results_df = evaluator.evaluate_all_models(predictions, y_test.values)
        evaluator.print_results(results_df)
        
        # Get best model
        best_model_name, best_metrics = evaluator.get_best_model(results_df)
        print(f"\n🏆 Best model: {best_model_name}")
        
        # Step 7: Save results
        print("\n[7/7] Saving results...")
        metrics_data = {
            'problem_type': 'timeseries',
            'best_model': best_model_name,
            'metrics': best_metrics,
            'all_results': results_df.to_dict('records'),
            'timeseries_info': ts_info
        }
        
        metrics_path = os.path.join(artifacts_path, 'timeseries_metrics.json')
        save_json(metrics_data, metrics_path)
        
        print("\n" + "="*80)
        print("TIME-SERIES AUTOML PIPELINE COMPLETED")
        print("="*80)
        print(f"\nArtifacts saved to: {artifacts_path}")
        
        return {
            'best_model_name': best_model_name,
            'best_metrics': best_metrics,
            'results_df': results_df,
            'predictions': predictions,
            'artifacts_path': artifacts_path
        }

    except Exception as e:
        logger.error(f"Unexpected error in Time-Series AutoML pipeline: {e}")
        logger.error(traceback.format_exc())
        print(f"\n✗ CRITICAL ERROR: {e}")
        print("\nFull traceback:")
        print(traceback.format_exc())
        raise RuntimeError(f"Time-Series AutoML pipeline failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Time-Series AutoML Pipeline')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--date-column', type=str, default=None, help='Date column name (optional)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length for DL models (default: 10)')
    
    args = parser.parse_args()
    
    run_timeseries_automl(
        csv_path=args.csv,
        target_column=args.target,
        date_column=args.date_column,
        test_size=args.test_size,
        sequence_length=args.sequence_length
    )
