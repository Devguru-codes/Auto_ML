"""
Model Zoo - Collection of ML models for classification and regression
"""
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from config.config import CLASSIFICATION_MODELS, REGRESSION_MODELS


class ModelZoo:
    """Factory for creating ML models"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type
        self.models = {}
        self.param_grids = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on problem type"""
        if self.problem_type == 'classification':
            self.models = {
                'logistic_regression': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'xgboost': XGBClassifier(random_state=42, eval_metric='logloss')
            }
            self.param_grids = {
                k: v['params'] for k, v in CLASSIFICATION_MODELS.items()
            }
        else:  # regression
            self.models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'xgboost': XGBRegressor(random_state=42)
            }
            self.param_grids = {
                k: v['params'] for k, v in REGRESSION_MODELS.items()
            }
        
        print(f"Initialized {len(self.models)} models for {self.problem_type}")
    
    def get_model(self, model_name: str):
        """Get a specific model by name"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        return self.models[model_name]
    
    def get_param_grid(self, model_name: str) -> dict:
        """Get parameter grid for a specific model"""
        if model_name not in self.param_grids:
            raise ValueError(f"Param grid for '{model_name}' not found")
        return self.param_grids[model_name]
    
    def get_all_models(self) -> dict:
        """Get all models"""
        return self.models
    
    def get_all_param_grids(self) -> dict:
        """Get all parameter grids"""
        return self.param_grids
    
    def get_model_display_name(self, model_name: str) -> str:
        """Get display name for model"""
        if self.problem_type == 'classification':
            return CLASSIFICATION_MODELS.get(model_name, {}).get('name', model_name)
        else:
            return REGRESSION_MODELS.get(model_name, {}).get('name', model_name)
