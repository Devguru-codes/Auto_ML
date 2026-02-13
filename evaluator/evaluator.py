"""
Model evaluation module
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Any


class ModelEvaluator:
    """Evaluate and compare multiple models"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type
        self.results = []
    
    def evaluate_classification(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Evaluate classification model"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # ROC AUC (handle binary and multiclass)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # Multiclass
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                metrics['roc_auc'] = None
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
            metrics['roc_auc'] = None
        
        return metrics
    
    def evaluate_regression(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Evaluate regression model"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, float]:
        """Evaluate a single model based on problem type"""
        if self.problem_type == 'classification':
            return self.evaluate_classification(model, X_test, y_test, model_name)
        else:
            return self.evaluate_regression(model, X_test, y_test, model_name)
    
    def evaluate_all_models(self, models: dict, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate all models and return results as DataFrame
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with evaluation metrics
        """
        self.results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            self.results.append(metrics)
        
        results_df = pd.DataFrame(self.results)
        
        # Sort by primary metric
        if self.problem_type == 'classification':
            results_df = results_df.sort_values('accuracy', ascending=False)
        else:
            results_df = results_df.sort_values('r2', ascending=False)
        
        return results_df
    
    def get_best_model(self, models: dict, results_df: pd.DataFrame) -> tuple:
        """
        Get the best performing model
        
        Returns:
            (best_model_name, best_model, best_metrics)
        """
        best_model_name = results_df.iloc[0]['model_name']
        best_model = models[best_model_name]
        best_metrics = results_df.iloc[0].to_dict()
        
        return best_model_name, best_model, best_metrics
    
    def print_results(self, results_df: pd.DataFrame):
        """Print formatted results"""
        print("\n" + "="*80)
        print("MODEL EVALUATION RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80 + "\n")
