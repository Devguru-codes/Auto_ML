"""
Configuration settings for AutoTabML
"""

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
