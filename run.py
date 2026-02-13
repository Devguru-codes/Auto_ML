"""
Main AutoML Pipeline Runner
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import validate_csv, validate_target_column, save_model, save_json, create_artifacts_dir
from utils.problem_detector import ProblemDetector
from preprocessing.preprocessor import AutoPreprocessor
from models.model_zoo import ModelZoo
from tuner.tuner import HyperparameterTuner
from evaluator.evaluator import ModelEvaluator
from explainability.shap_explainer import ShapExplainer
from config.config import (
    ARTIFACTS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME, 
    METRICS_FILENAME, SHAP_PLOT_FILENAME, FEATURE_IMPORTANCE_FILENAME,
    CLASSIFICATION_METRICS, REGRESSION_METRICS
)


def run_automl(csv_path: str, target_column: str, metric: str = None, test_size: float = 0.2):
    """
    Run complete AutoML pipeline
    
    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
        metric: Optional evaluation metric (auto-selected if None)
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary with results
    """
    print("="*80)
    print("AUTOML PIPELINE STARTED")
    print("="*80)
    
    # Create artifacts directory
    artifacts_path = create_artifacts_dir(ARTIFACTS_DIR)
    
    # Step 1: Load and validate data
    print("\n[1/8] Loading and validating data...")
    df = validate_csv(csv_path)
    validate_target_column(df, target_column)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    
    # Step 2: Detect problem type
    print("\n[2/8] Detecting problem type...")
    detector = ProblemDetector()
    problem_type = detector.detect(df[target_column])
    problem_info = detector.get_problem_info()
    
    # Step 3: Prepare data
    print("\n[3/8] Preparing data...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if problem_type == 'classification' else None
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 4: Build preprocessing pipeline
    print("\n[4/8] Building preprocessing pipeline...")
    preprocessor = AutoPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed shape: {X_train_processed.shape}")
    
    # Step 5: Initialize model zoo
    print("\n[5/8] Initializing model zoo...")
    model_zoo = ModelZoo(problem_type)
    models = model_zoo.get_all_models()
    param_grids = model_zoo.get_all_param_grids()
    
    # Determine scoring metric
    if metric is None:
        if problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
    else:
        scoring = metric
    
    print(f"Using scoring metric: {scoring}")
    
    # Step 6: Hyperparameter tuning
    print("\n[6/8] Tuning hyperparameters...")
    tuner = HyperparameterTuner()
    tuned_models = tuner.tune_all_models(models, param_grids, X_train_processed, y_train, scoring)
    
    # Step 7: Evaluate models
    print("\n[7/8] Evaluating models...")
    evaluator = ModelEvaluator(problem_type)
    results_df = evaluator.evaluate_all_models(tuned_models, X_test_processed, y_test)
    evaluator.print_results(results_df)
    
    # Get best model
    best_model_name, best_model, best_metrics = evaluator.get_best_model(tuned_models, results_df)
    print(f"\nBest model: {best_model_name}")
    
    # Step 8: Generate explanations
    print("\n[8/8] Generating explanations...")
    try:
        explainer = ShapExplainer(
            best_model, 
            X_train_processed,
            feature_names=preprocessor.get_feature_names()
        )
        
        shap_plot_path = os.path.join(artifacts_path, SHAP_PLOT_FILENAME)
        feature_importance_path = os.path.join(artifacts_path, FEATURE_IMPORTANCE_FILENAME)
        
        explainer.plot_summary(X_test_processed, save_path=shap_plot_path)
        explainer.plot_feature_importance(X_test_processed, save_path=feature_importance_path)
    except Exception as e:
        print(f"Warning: Could not generate SHAP explanations: {e}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    model_path = os.path.join(artifacts_path, MODEL_FILENAME)
    preprocessor_path = os.path.join(artifacts_path, PREPROCESSOR_FILENAME)
    metrics_path = os.path.join(artifacts_path, METRICS_FILENAME)
    
    save_model(best_model, model_path)
    save_model(preprocessor, preprocessor_path)
    
    # Prepare metrics for saving
    metrics_data = {
        'problem_type': problem_type,
        'best_model': best_model_name,
        'metrics': best_metrics,
        'all_results': results_df.to_dict('records')
    }
    save_json(metrics_data, metrics_path)
    
    print("\n" + "="*80)
    print("AUTOML PIPELINE COMPLETED")
    print("="*80)
    print(f"\nArtifacts saved to: {artifacts_path}")
    print(f"  - Model: {MODEL_FILENAME}")
    print(f"  - Preprocessor: {PREPROCESSOR_FILENAME}")
    print(f"  - Metrics: {METRICS_FILENAME}")
    print(f"  - SHAP plots: {SHAP_PLOT_FILENAME}, {FEATURE_IMPORTANCE_FILENAME}")
    
    return {
        'problem_type': problem_type,
        'best_model_name': best_model_name,
        'best_metrics': best_metrics,
        'results_df': results_df,
        'artifacts_path': artifacts_path
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoML Pipeline')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--metric', type=str, default=None, help='Evaluation metric (optional)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    run_automl(
        csv_path=args.csv,
        target_column=args.target,
        metric=args.metric,
        test_size=args.test_size
    )
