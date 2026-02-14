"""
Optuna-based hyperparameter tuning for better optimization
"""
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import numpy as np
from typing import Dict, Any


class OptunaHyperparameterTuner:
    """Hyperparameter tuning using Optuna (Bayesian optimization)"""
    
    def __init__(self, n_trials: int = 50, timeout: int = 600, cv: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.random_state = random_state
        self.best_estimators = {}
        self.studies = {}
    
    def _create_objective(self, model, param_grid: dict, X, y, scoring: str):
        """
        Create Optuna objective function
        
        Args:
            model: Sklearn model instance
            param_grid: Dictionary of parameters to search
            X: Training features
            y: Training target
            scoring: Scoring metric
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Sample hyperparameters based on their type
            params = {}
            
            for param_name, param_values in param_grid.items():
                # Handle nested parameters (e.g., model__learning_rate for scikeras)
                clean_param_name = param_name.replace('model__', '')
                
                if isinstance(param_values, list):
                    if len(param_values) == 0:
                        continue
                    
                    # Check if all values are integers
                    if all(isinstance(x, int) for x in param_values):
                        params[param_name] = trial.suggest_int(
                            clean_param_name, 
                            min(param_values), 
                            max(param_values)
                        )
                    # Check if all values are floats
                    elif all(isinstance(x, (int, float)) for x in param_values):
                        params[param_name] = trial.suggest_float(
                            clean_param_name,
                            min(param_values),
                            max(param_values),
                            log=True if min(param_values) > 0 and max(param_values)/min(param_values) > 100 else False
                        )
                    # Categorical
                    else:
                        params[param_name] = trial.suggest_categorical(clean_param_name, param_values)
            
            # Set parameters
            try:
                model.set_params(**params)
            except Exception as e:
                print(f"Warning: Could not set params {params}: {e}")
                return -np.inf
            
            # Cross-validation
            try:
                scores = cross_val_score(
                    model, X, y, 
                    cv=self.cv, 
                    scoring=scoring,
                    n_jobs=-1
                )
                return scores.mean()
            except Exception as e:
                print(f"Warning: CV failed with params {params}: {e}")
                return -np.inf
        
        return objective
    
    def tune_model(self, model, param_grid: dict, X, y, scoring: str):
        """
        Tune hyperparameters for a single model using Optuna
        
        Args:
            model: Sklearn model instance
            param_grid: Dictionary of parameters to search
            X: Training features
            y: Training target
            scoring: Scoring metric
            
        Returns:
            Best estimator after tuning
        """
        model_name = model.__class__.__name__
        print(f"Tuning {model_name} with Optuna...")
        
        # Handle empty param grids
        if not param_grid or len(param_grid) == 0:
            print(f"No parameters to tune for {model_name}, using default")
            model.fit(X, y)
            return model
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        # Create objective
        objective = self._create_objective(model, param_grid, X, y, scoring)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
            n_jobs=1  # Parallel trials can cause issues with sklearn
        )
        
        # Get best parameters
        best_params = study.best_params
        
        # Convert back parameter names (remove clean names)
        final_params = {}
        for param_name in param_grid.keys():
            clean_name = param_name.replace('model__', '')
            if clean_name in best_params:
                final_params[param_name] = best_params[clean_name]
        
        print(f"Best parameters: {final_params}")
        print(f"Best CV score: {study.best_value:.4f}")
        print(f"Number of trials: {len(study.trials)}")
        
        # Train final model with best parameters
        model.set_params(**final_params)
        model.fit(X, y)
        
        # Store study for later analysis
        self.studies[model_name] = study
        
        return model
    
    def tune_all_models(self, models: dict, param_grids: dict, X, y, scoring: str) -> dict:
        """
        Tune hyperparameters for all models using Optuna
        
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
    
    def plot_optimization_history(self, model_name: str, save_path: str = None):
        """Plot optimization history for a model"""
        if model_name not in self.studies:
            print(f"No study found for {model_name}")
            return
        
        study = self.studies[model_name]
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Optimization history saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Could not plot optimization history: {e}")
    
    def plot_param_importances(self, model_name: str, save_path: str = None):
        """Plot parameter importances for a model"""
        if model_name not in self.studies:
            print(f"No study found for {model_name}")
            return
        
        study = self.studies[model_name]
        
        try:
            import matplotlib.pyplot as plt
            
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Parameter importances saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
