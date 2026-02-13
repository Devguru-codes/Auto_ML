"""
Problem type detector - automatically determines if problem is classification or regression
"""
import pandas as pd
import numpy as np
from config.config import REGRESSION_THRESHOLD


class ProblemDetector:
    """Automatically detect problem type (classification or regression)"""
    
    def __init__(self, threshold: int = REGRESSION_THRESHOLD):
        self.threshold = threshold
        self.problem_type = None
        self.num_classes = None
    
    def detect(self, y: pd.Series) -> str:
        """
        Detect problem type based on target variable
        
        Args:
            y: Target variable series
            
        Returns:
            'classification' or 'regression'
        """
        # Remove missing values for analysis
        y_clean = y.dropna()
        
        if len(y_clean) == 0:
            raise ValueError("Target variable has no valid values")
        
        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(y_clean)
        
        # Count unique values
        unique_values = y_clean.nunique()
        
        # Decision logic
        if is_numeric and unique_values > self.threshold:
            self.problem_type = 'regression'
            self.num_classes = None
        else:
            self.problem_type = 'classification'
            self.num_classes = unique_values
        
        print(f"Problem type detected: {self.problem_type}")
        if self.problem_type == 'classification':
            print(f"Number of classes: {self.num_classes}")
        
        return self.problem_type
    
    def get_problem_info(self) -> dict:
        """Get problem type information"""
        return {
            'problem_type': self.problem_type,
            'num_classes': self.num_classes
        }
