"""
Time-series preprocessing module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


class TimeSeriesPreprocessor:
    """Preprocessing for time-series data"""
    
    def __init__(self, sequence_length: int = 10, forecast_horizon: int = 1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.is_stationary = False
        self.diff_order = 0
    
    def check_stationarity(self, series: pd.Series, significance_level: float = 0.05) -> bool:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series data
            significance_level: P-value threshold
            
        Returns:
            True if stationary, False otherwise
        """
        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            self.is_stationary = p_value < significance_level
            
            print(f"ADF Statistic: {result[0]:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Stationary: {self.is_stationary}")
            
            return self.is_stationary
        except Exception as e:
            print(f"Warning: Could not perform stationarity test: {e}")
            return False
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> pd.Series:
        """
        Make time series stationary through differencing
        
        Args:
            series: Time series data
            max_diff: Maximum differencing order
            
        Returns:
            Stationary series
        """
        stationary_series = series.copy()
        
        for d in range(1, max_diff + 1):
            if self.check_stationarity(stationary_series):
                self.diff_order = d - 1
                break
            stationary_series = stationary_series.diff().dropna()
            self.diff_order = d
        
        return stationary_series
    
    def create_sequences(self, data: np.ndarray, include_target: bool = True):
        """
        Create sequences for deep learning models (LSTM, GRU)
        
        Args:
            data: Time series data as numpy array
            include_target: Whether to create target sequences
            
        Returns:
            X, y arrays for training
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i:i + self.sequence_length])
            
            # Target (next values)
            if include_target:
                if self.forecast_horizon == 1:
                    y.append(data[i + self.sequence_length])
                else:
                    y.append(data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def scale_data(self, data: np.ndarray, fit: bool = True):
        """
        Scale data using StandardScaler
        
        Args:
            data: Data to scale
            fit: Whether to fit the scaler
            
        Returns:
            Scaled data
        """
        if fit:
            return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            return self.scaler.transform(data.reshape(-1, 1)).flatten()
    
    def inverse_scale(self, data: np.ndarray):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    def train_test_split_temporal(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Split time series data maintaining temporal order
        
        Args:
            df: Time series dataframe
            test_size: Fraction for test set
            
        Returns:
            train_df, test_df
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return train_df, test_df
    
    def prepare_for_arima(self, series: pd.Series):
        """
        Prepare data for ARIMA modeling
        
        Args:
            series: Time series data
            
        Returns:
            Prepared series
        """
        # Check and make stationary if needed
        if not self.check_stationarity(series):
            print("Series is not stationary. Applying differencing...")
            series = self.make_stationary(series)
        
        return series
    
    def prepare_for_prophet(self, df: pd.DataFrame, target_column: str):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            df: Dataframe with datetime index
            target_column: Target column name
            
        Returns:
            Prophet-formatted dataframe
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_column].values
        })
        
        return prophet_df
    
    def prepare_for_deep_learning(self, series: pd.Series, scale: bool = True):
        """
        Prepare data for deep learning models (LSTM, GRU)
        
        Args:
            series: Time series data
            scale: Whether to scale the data
            
        Returns:
            X, y sequences
        """
        # Convert to numpy
        data = series.values
        
        # Scale if requested
        if scale:
            data = self.scale_data(data, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(data)
        
        # Reshape for LSTM/GRU (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
