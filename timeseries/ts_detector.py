"""
Time-series detection module
"""
import pandas as pd
import numpy as np


class TimeSeriesDetector:
    """Detect if dataset is time-series and extract temporal information"""
    
    def __init__(self):
        self.is_timeseries = False
        self.date_column = None
        self.frequency = None
        self.has_seasonality = False
    
    def detect(self, df: pd.DataFrame, date_column: str = None, target_column: str = None) -> bool:
        """
        Detect if dataset is time-series
        
        Args:
            df: Input dataframe
            date_column: Optional explicit date column name
            target_column: Target column name
            
        Returns:
            True if time-series, False otherwise
        """
        # Method 1: User specified date column
        if date_column and date_column in df.columns:
            self.date_column = date_column
            self.is_timeseries = True
            self._analyze_temporal_patterns(df, date_column, target_column)
            return True
        
        # Method 2: Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if len(datetime_cols) > 0:
            self.date_column = datetime_cols[0]
            self.is_timeseries = True
            self._analyze_temporal_patterns(df, self.date_column, target_column)
            return True
        
        # Method 3: Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            self.date_column = df.index.name or 'index'
            self.is_timeseries = True
            self._analyze_temporal_patterns(df, None, target_column)  # Use index
            return True
        
        # Method 4: Try to parse potential date columns
        for col in df.columns:
            if col == target_column:
                continue
            
            # Check if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                try:
                    pd.to_datetime(df[col])
                    self.date_column = col
                    self.is_timeseries = True
                    self._analyze_temporal_patterns(df, col, target_column)
                    return True
                except:
                    continue
        
        self.is_timeseries = False
        return False
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, date_column: str = None, target_column: str = None):
        """Analyze temporal patterns in the data"""
        try:
            # Get datetime series
            if date_column:
                dt_series = pd.to_datetime(df[date_column])
            else:
                dt_series = df.index
            
            # Infer frequency
            if len(dt_series) > 1:
                time_diffs = dt_series.diff().dropna()
                most_common_diff = time_diffs.mode()[0] if len(time_diffs) > 0 else None
                
                if most_common_diff:
                    # Map to pandas frequency strings
                    days = most_common_diff.days
                    if days == 1:
                        self.frequency = 'D'  # Daily
                    elif days == 7:
                        self.frequency = 'W'  # Weekly
                    elif 28 <= days <= 31:
                        self.frequency = 'M'  # Monthly
                    elif 365 <= days <= 366:
                        self.frequency = 'Y'  # Yearly
                    else:
                        self.frequency = f'{days}D'
            
            # Check for seasonality (simple heuristic)
            if target_column and target_column in df.columns:
                # If we have enough data points, check for repeating patterns
                if len(df) >= 24:  # At least 2 years of monthly data
                    self.has_seasonality = True  # Assume seasonality for now
        
        except Exception as e:
            print(f"Warning: Could not analyze temporal patterns: {e}")
    
    def get_info(self) -> dict:
        """Get time-series information"""
        return {
            'is_timeseries': self.is_timeseries,
            'date_column': self.date_column,
            'frequency': self.frequency,
            'has_seasonality': self.has_seasonality
        }
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_column: str):
        """
        Prepare dataframe for time-series modeling
        
        Args:
            df: Input dataframe
            target_column: Target column name
            
        Returns:
            Prepared dataframe with datetime index
        """
        if not self.is_timeseries:
            raise ValueError("Dataset is not time-series")
        
        df_copy = df.copy()
        
        # Set datetime index
        if self.date_column and self.date_column in df_copy.columns:
            df_copy[self.date_column] = pd.to_datetime(df_copy[self.date_column])
            df_copy = df_copy.set_index(self.date_column)
        
        # Sort by index
        df_copy = df_copy.sort_index()
        
        return df_copy
