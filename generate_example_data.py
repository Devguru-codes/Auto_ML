"""
Generate example datasets for testing AutoML
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Generate classification dataset (Iris-like)
print("Generating classification dataset...")
X_class, y_class = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=7,
    n_redundant=2,
    n_classes=3,
    random_state=42
)

# Create feature names
feature_names = [f'feature_{i}' for i in range(10)]

# Add some categorical features
df_class = pd.DataFrame(X_class, columns=feature_names)
df_class['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=500)
df_class['category_B'] = np.random.choice(['Low', 'Medium', 'High'], size=500)
df_class['target'] = y_class

# Save classification dataset
df_class.to_csv('data/classification_example.csv', index=False)
print(f"Classification dataset saved: {df_class.shape}")
print(f"Classes: {df_class['target'].unique()}")

# Generate regression dataset (House prices-like)
print("\nGenerating regression dataset...")
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=8,
    n_informative=6,
    noise=10,
    random_state=42
)

# Create feature names
reg_feature_names = ['area', 'bedrooms', 'bathrooms', 'age', 'distance_city', 
                     'schools_nearby', 'crime_rate', 'property_tax']

df_reg = pd.DataFrame(X_reg, columns=reg_feature_names)

# Add categorical features
df_reg['neighborhood'] = np.random.choice(['Downtown', 'Suburb', 'Rural'], size=500)
df_reg['house_type'] = np.random.choice(['Apartment', 'House', 'Condo'], size=500)

# Scale target to realistic house prices
df_reg['price'] = (y_reg - y_reg.min()) * 1000 + 100000

# Save regression dataset
df_reg.to_csv('data/regression_example.csv', index=False)
print(f"Regression dataset saved: {df_reg.shape}")
print(f"Price range: ${df_reg['price'].min():.2f} - ${df_reg['price'].max():.2f}")

print("\nDatasets created successfully!")
print("  - data/classification_example.csv")
print("  - data/regression_example.csv")
