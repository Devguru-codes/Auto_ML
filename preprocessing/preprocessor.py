"""
Preprocessing module for automatic feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class AutoPreprocessor:
    """Automatic preprocessing pipeline builder"""
    
    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
    
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build preprocessing pipeline based on data types
        
        Args:
            X: Feature dataframe
            
        Returns:
            ColumnTransformer pipeline
        """
        # Identify numeric and categorical columns
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        # Numeric pipeline: impute with mean, then scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute with most frequent, then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(('cat', categorical_transformer, self.categorical_features))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop any columns not specified
        )
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Build and fit preprocessor, then transform data"""
        if self.preprocessor is None:
            self.build_preprocessor(X)
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.preprocessor.transform(X)
    
    def get_feature_names(self) -> list:
        """Get feature names after transformation"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        
        # Get feature names from each transformer
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Get one-hot encoded feature names
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_features)
        
        return feature_names
