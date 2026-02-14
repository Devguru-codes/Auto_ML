"""
Model Zoo - Collection of ML models for classification and regression
"""
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from config.config import CLASSIFICATION_MODELS, REGRESSION_MODELS, NEURAL_NET_THRESHOLD

# Neural network imports
try:
    from scikeras.wrappers import KerasClassifier, KerasRegressor
    import tensorflow as tf
    from tensorflow import keras
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    print("Warning: TensorFlow/scikeras not available. Neural networks will be disabled.")


def create_nn_classifier(hidden_layer_sizes=(64, 32), learning_rate=0.001, input_dim=10):
    """Create a neural network classifier"""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Add hidden layers
    for units in hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
    
    # Output layer (will be configured by KerasClassifier)
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_nn_regressor(hidden_layer_sizes=(64, 32), learning_rate=0.001, input_dim=10):
    """Create a neural network regressor"""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Add hidden layers
    for units in hidden_layer_sizes:
        model.add(keras.layers.Dense(units, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
    
    # Output layer
    model.add(keras.layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model


class ModelZoo:
    """Factory for creating ML models"""
    
    def __init__(self, problem_type: str, n_samples: int = 0, input_dim: int = 10):
        self.problem_type = problem_type
        self.n_samples = n_samples
        self.input_dim = input_dim
        self.models = {}
        self.param_grids = {}
        self.use_neural_net = n_samples >= NEURAL_NET_THRESHOLD and NEURAL_NET_AVAILABLE
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
                k: v['params'] for k, v in CLASSIFICATION_MODELS.items() if k != 'neural_network'
            }
            
            # Add neural network if dataset is large enough
            print(f"Debug: n_samples={self.n_samples}, threshold={NEURAL_NET_THRESHOLD}, available={NEURAL_NET_AVAILABLE}, use_nn={self.use_neural_net}")
            if self.use_neural_net:
                print(f"✓ Dataset size ({self.n_samples}) >= threshold ({NEURAL_NET_THRESHOLD}). Adding Neural Network.")
                self.models['neural_network'] = KerasClassifier(
                    model=create_nn_classifier,
                    hidden_layer_sizes=(64, 32),
                    learning_rate=0.001,
                    input_dim=self.input_dim,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    random_state=42
                )
                self.param_grids['neural_network'] = CLASSIFICATION_MODELS['neural_network']['params']
            else:
                print(f"✗ Neural network not added (n_samples={self.n_samples} < {NEURAL_NET_THRESHOLD} or TF not available)")
            
        else:  # regression
            self.models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'xgboost': XGBRegressor(random_state=42)
            }
            self.param_grids = {
                k: v['params'] for k, v in REGRESSION_MODELS.items() if k != 'neural_network'
            }
            
            # Add neural network if dataset is large enough
            if self.use_neural_net:
                print(f"Dataset size ({self.n_samples}) >= threshold ({NEURAL_NET_THRESHOLD}). Adding Neural Network.")
                self.models['neural_network'] = KerasRegressor(
                    model=create_nn_regressor,
                    hidden_layer_sizes=(64, 32),
                    learning_rate=0.001,
                    input_dim=self.input_dim,
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    random_state=42
                )
                self.param_grids['neural_network'] = REGRESSION_MODELS['neural_network']['params']
        
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
