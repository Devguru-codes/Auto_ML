"""
Configuration settings for AutoTabML
"""

# Neural Network Threshold
# If dataset has >= this many samples, include neural network models
NEURAL_NET_THRESHOLD = 1000

# Model Zoo Configuration
CLASSIFICATION_MODELS = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [100, 200, 500],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'random_forest': {
        'name': 'Random Forest',
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'xgboost': {
        'name': 'XGBoost',
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'neural_network': {
        'name': 'Neural Network',
        'params': {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'model__learning_rate': [0.001, 0.01],
            'model__batch_size': [32, 64],
            'model__epochs': [50, 100]
        }
    }
}

REGRESSION_MODELS = {
    'linear_regression': {
        'name': 'Linear Regression',
        'params': {
            'fit_intercept': [True, False]
        }
    },
    'random_forest': {
        'name': 'Random Forest Regressor',
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting Regressor',
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'xgboost': {
        'name': 'XGBoost Regressor',
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'neural_network': {
        'name': 'Neural Network Regressor',
        'params': {
            'model__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
            'model__learning_rate': [0.001, 0.01],
            'model__batch_size': [32, 64],
            'model__epochs': [50, 100]
        }
    }
}

# Tuning Configuration
TUNING_CONFIG = {
    'n_iter': 20,  # Number of parameter settings sampled
    'cv': 5,  # Cross-validation folds
    'random_state': 42,
    'n_jobs': -1,  # Use all available cores
    'verbose': 1
}

# Problem Detection Thresholds
REGRESSION_THRESHOLD = 20  # If unique values > threshold, treat as regression

# Metrics
CLASSIFICATION_METRICS = ['accuracy', 'f1_weighted', 'roc_auc_ovr']
REGRESSION_METRICS = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

# Artifacts
ARTIFACTS_DIR = 'artifacts'
MODEL_FILENAME = 'best_model.pkl'
PREPROCESSOR_FILENAME = 'preprocessor.pkl'
METRICS_FILENAME = 'metrics.json'
SHAP_PLOT_FILENAME = 'shap_summary.png'
FEATURE_IMPORTANCE_FILENAME = 'feature_importance.png'

# Visualization artifacts
MODEL_COMPARISON_FILENAME = 'model_comparison.png'
CONFUSION_MATRIX_FILENAME = 'confusion_matrix.png'
ROC_CURVES_FILENAME = 'roc_curves.png'
TRAINING_HISTORY_FILENAME = 'training_history.png'
METRICS_SUMMARY_FILENAME = 'metrics_summary_dashboard.png'

# Optuna Configuration
OPTUNA_CONFIG = {
    'n_trials': 50,
    'timeout': 600,  # 10 minutes
    'cv': 5,
    'random_state': 42
}

# Time-Series Model Configuration
TIMESERIES_MODELS = {
    'arima': {
        'name': 'Auto ARIMA',
        'params': {
            'seasonal': [True, False],
            'm': [1, 7, 12],  # Daily, Weekly, Monthly seasonality
            'max_p': [3, 5],
            'max_q': [3, 5],
            'max_d': [1, 2]
        }
    },
    'prophet': {
        'name': 'Prophet',
        'params': {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False]
        }
    },
    'lstm': {
        'name': 'LSTM',
        'params': {
            'units': [32, 64, 128],
            'layers': [1, 2, 3],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01],
            'epochs': [50, 100],
            'batch_size': [16, 32, 64],
            'sequence_length': [10, 20, 30]
        }
    },
    'gru': {
        'name': 'GRU',
        'params': {
            'units': [32, 64, 128],
            'layers': [1, 2, 3],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01],
            'epochs': [50, 100],
            'batch_size': [16, 32, 64],
            'sequence_length': [10, 20, 30]
        }
    },
    'bidirectional_lstm': {
        'name': 'Bidirectional LSTM',
        'params': {
            'units': [32, 64, 128],
            'layers': [1, 2],
            'dropout': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.01],
            'epochs': [50, 100],
            'batch_size': [16, 32, 64],
            'sequence_length': [10, 20, 30]
        }
    }
}

# Time-Series Metrics
TIMESERIES_METRICS = ['mae', 'rmse', 'mape', 'smape', 'r2']
