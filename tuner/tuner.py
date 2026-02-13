"""
Hyperparameter tuning module using RandomizedSearchCV
"""
from sklearn.model_selection import RandomizedSearchCV
from config.config import TUNING_CONFIG
import numpy as np


class HyperparameterTuner:
    """Hyperparameter tuning using RandomizedSearchCV"""
    
    def __init__(self, n_iter: int = None, cv: int = None, random_state: int = 42):
        self.n_iter = n_iter or TUNING_CONFIG['n_iter']
        self.cv = cv or TUNING_CONFIG['cv']
        self.random_state = random_state
        self.n_jobs = TUNING_CONFIG['n_jobs']
        self.verbose = TUNING_CONFIG['verbose']
        self.best_estimators = {}
    
    def tune_model(self, model, param_grid: dict, X, y, scoring: str):
        """
        Tune hyperparameters for a single model
        
        Args:
            model: Sklearn model instance
            param_grid: Dictionary of parameters to search
            X: Training features
            y: Training target
            scoring: Scoring metric
            
        Returns:
            Best estimator after tuning
        """
        print(f"Tuning {model.__class__.__name__}...")
        
        # Handle empty param grids
        if not param_grid or len(param_grid) == 0:
            print(f"No parameters to tune for {model.__class__.__name__}, using default")
            model.fit(X, y)
            return model
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def tune_all_models(self, models: dict, param_grids: dict, X, y, scoring: str) -> dict:
        """
        Tune hyperparameters for all models
        
        Args:
            models: Dictionary of model instances
            param_grids: Dictionary of parameter grids
            X: Training features
            y: Training target
            scoring: Scoring metric
            
        Returns:
            Dictionary of best estimators
        """
        self.best_estimators = {}
        
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            best_model = self.tune_model(model, param_grid, X, y, scoring)
            self.best_estimators[model_name] = best_model
            print(f"Completed tuning for {model_name}\n")
        
        return self.best_estimators
