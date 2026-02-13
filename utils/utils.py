"""
Utility functions for AutoTabML
"""
import os
import json
import joblib
from typing import Any, Dict
import pandas as pd


def save_model(model: Any, filepath: str) -> None:
    """Save model to disk using joblib"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load model from disk"""
    return joblib.load(filepath)


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_csv(filepath: str) -> pd.DataFrame:
    """Validate and load CSV file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.endswith('.csv'):
        raise ValueError("File must be a CSV file")
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {str(e)}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate target column exists in dataframe"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}")
    
    if df[target_column].isna().all():
        raise ValueError(f"Target column '{target_column}' contains only missing values")


def create_artifacts_dir(base_dir: str = 'artifacts') -> str:
    """Create artifacts directory if it doesn't exist"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir
