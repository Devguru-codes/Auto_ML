"""
Time-series models: ARIMA, Prophet, LSTM, GRU, Bidirectional LSTM
"""
import numpy as np
import pandas as pd
from typing import Tuple

# Statistical models
try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: pmdarima not available (ARIMA model disabled)")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: prophet not available (Prophet model disabled)")

# Deep learning models
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Warning: TensorFlow not available")


class ARIMAModel:
    """Auto ARIMA model using pmdarima"""
    
    def __init__(self, seasonal=True, m=12, max_p=5, max_q=5, max_d=2):
        self.seasonal = seasonal
        self.m = m  # Seasonality period
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.model = None
    
    def fit(self, y_train):
        """Fit ARIMA model"""
        if not ARIMA_AVAILABLE:
            raise ImportError("pmdarima not installed")
        
        self.model = auto_arima(
            y_train,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        return self
    
    def predict(self, n_periods=1):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(n_periods=n_periods)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'seasonal': self.seasonal,
            'm': self.m,
            'max_p': self.max_p,
            'max_q': self.max_q,
            'max_d': self.max_d
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ProphetModel:
    """Facebook Prophet model"""
    
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, 
                 seasonality_mode='additive', yearly_seasonality=True, 
                 weekly_seasonality=True, daily_seasonality=False):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
    
    def fit(self, df):
        """
        Fit Prophet model
        
        Args:
            df: DataFrame with 'ds' (datetime) and 'y' (target) columns
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet not installed")
        
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        
        self.model.fit(df)
        return self
    
    def predict(self, periods=1, freq='D'):
        """Make future predictions"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_params(self, deep=True):
        """Get parameters"""
        return {
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality
        }
    
    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LSTMModel:
    """LSTM model for time series forecasting"""
    
    def __init__(self, units=64, layers=2, dropout=0.2, learning_rate=0.001, 
                 epochs=50, batch_size=32, sequence_length=10):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """Build LSTM architecture"""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.units,
            return_sequences=True if self.layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.layers):
            return_seq = i < self.layers - 1
            model.add(LSTM(self.units, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        if self.model is None:
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                                   patience=10, restore_best_weights=True)
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X, verbose=0)
    
    def get_params(self, deep=True):
        """Get parameters"""
        return {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length
        }
    
    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        self.model = None  # Reset model when params change
        return self


class GRUModel:
    """GRU model for time series forecasting (faster than LSTM)"""
    
    def __init__(self, units=64, layers=2, dropout=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, sequence_length=10):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """Build GRU architecture"""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(
            self.units,
            return_sequences=True if self.layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Additional GRU layers
        for i in range(1, self.layers):
            return_seq = i < self.layers - 1
            model.add(GRU(self.units, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train GRU model"""
        if self.model is None:
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                                   patience=10, restore_best_weights=True)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X, verbose=0)
    
    def get_params(self, deep=True):
        """Get parameters"""
        return {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length
        }
    
    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        self.model = None
        return self


class BidirectionalLSTMModel:
    """Bidirectional LSTM model (processes sequence in both directions)"""
    
    def __init__(self, units=64, layers=2, dropout=0.2, learning_rate=0.001,
                 epochs=50, batch_size=32, sequence_length=10):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """Build Bidirectional LSTM architecture"""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow not installed")
        
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(self.units, return_sequences=True if self.layers > 1 else False),
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout))
        
        # Additional Bidirectional LSTM layers
        for i in range(1, self.layers):
            return_seq = i < self.layers - 1
            model.add(Bidirectional(LSTM(self.units, return_sequences=return_seq)))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train Bidirectional LSTM model"""
        if self.model is None:
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                                   patience=10, restore_best_weights=True)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return self.model.predict(X, verbose=0)
    
    def get_params(self, deep=True):
        """Get parameters"""
        return {
            'units': self.units,
            'layers': self.layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length
        }
    
    def set_params(self, **params):
        """Set parameters"""
        for key, value in params.items():
            setattr(self, key, value)
        self.model = None
        return self


class TimeSeriesModelZoo:
    """Factory for time-series models"""
    
    def __init__(self, model_types=['arima', 'prophet', 'lstm', 'gru', 'bidirectional_lstm']):
        self.model_types = model_types
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize time-series models"""
        if 'arima' in self.model_types and ARIMA_AVAILABLE:
            self.models['arima'] = ARIMAModel()
        
        if 'prophet' in self.model_types and PROPHET_AVAILABLE:
            self.models['prophet'] = ProphetModel()
        
        if 'lstm' in self.model_types and DEEP_LEARNING_AVAILABLE:
            self.models['lstm'] = LSTMModel()
        
        if 'gru' in self.model_types and DEEP_LEARNING_AVAILABLE:
            self.models['gru'] = GRUModel()
        
        if 'bidirectional_lstm' in self.model_types and DEEP_LEARNING_AVAILABLE:
            self.models['bidirectional_lstm'] = BidirectionalLSTMModel()
        
        print(f"Initialized {len(self.models)} time-series models: {list(self.models.keys())}")
    
    def get_all_models(self):
        """Get all models"""
        return self.models
    
    def get_model(self, model_name):
        """Get specific model"""
        return self.models.get(model_name)
