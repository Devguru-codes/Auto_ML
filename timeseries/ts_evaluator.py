"""
Time-series model evaluation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesEvaluator:
    """Evaluate time-series forecasting models"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def evaluate_model(self, y_true, y_pred, model_name: str):
        """
        Evaluate a single model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Flatten if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        metrics = {
            'model_name': model_name,
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': self.calculate_mape(y_true, y_pred),
            'smape': self.calculate_smape(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, models_predictions: dict, y_true):
        """
        Evaluate all models
        
        Args:
            models_predictions: Dictionary of {model_name: predictions}
            y_true: True values
            
        Returns:
            DataFrame with results
        """
        results_list = []
        
        for model_name, y_pred in models_predictions.items():
            metrics = self.evaluate_model(y_true, y_pred, model_name)
            results_list.append(metrics)
        
        results_df = pd.DataFrame(results_list)
        
        # Sort by RMSE (lower is better)
        results_df = results_df.sort_values('rmse', ascending=True).reset_index(drop=True)
        
        return results_df
    
    def print_results(self, results_df: pd.DataFrame):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("TIME-SERIES MODEL EVALUATION RESULTS")
        print("="*80)
        
        for idx, row in results_df.iterrows():
            print(f"\n{idx + 1}. {row['model_name'].upper()}")
            print(f"   MAE:   {row['mae']:.4f}")
            print(f"   RMSE:  {row['rmse']:.4f}")
            print(f"   MAPE:  {row['mape']:.2f}%")
            print(f"   SMAPE: {row['smape']:.2f}%")
            print(f"   R²:    {row['r2']:.4f}")
        
        print("\n" + "="*80)
    
    def get_best_model(self, results_df: pd.DataFrame):
        """Get best model based on RMSE"""
        best_row = results_df.iloc[0]
        return best_row['model_name'], best_row.to_dict()
