"""
Test AutoML with sklearn datasets (MNIST-like digits dataset)
"""
import pandas as pd
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
import os

# Create data directory
os.makedirs('data', exist_ok=True)

print("="*60)
print("GENERATING SKLEARN DATASETS FOR TESTING")
print("="*60)

# 1. DIGITS DATASET (MNIST-like, 1797 samples - will trigger neural network!)
print("\n[1/3] Loading Digits dataset (like MNIST)...")
digits = load_digits()
df_digits = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
df_digits['target'] = digits.target

print(f"  Shape: {df_digits.shape}")
print(f"  Classes: {len(digits.target_names)} (digits 0-9)")
print(f"  Features: 64 pixels (8x8 images)")
print(f"  Samples: {len(df_digits)} - WILL TRIGGER NEURAL NETWORK (>= 1000)")

df_digits.to_csv('data/sklearn_digits.csv', index=False)
print(f"  ✓ Saved to data/sklearn_digits.csv")

# 2. WINE DATASET (178 samples - traditional ML only)
print("\n[2/3] Loading Wine dataset...")
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine['target'] = wine.target

print(f"  Shape: {df_wine.shape}")
print(f"  Classes: {len(wine.target_names)}")
print(f"  Samples: {len(df_wine)} - traditional ML only (< 1000)")

df_wine.to_csv('data/sklearn_wine.csv', index=False)
print(f"  ✓ Saved to data/sklearn_wine.csv")

# 3. BREAST CANCER DATASET (569 samples - traditional ML only)
print("\n[3/3] Loading Breast Cancer dataset...")
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df_cancer['target'] = cancer.target

print(f"  Shape: {df_cancer.shape}")
print(f"  Classes: {len(cancer.target_names)} (malignant/benign)")
print(f"  Samples: {len(df_cancer)} - traditional ML only (< 1000)")

df_cancer.to_csv('data/sklearn_breast_cancer.csv', index=False)
print(f"  ✓ Saved to data/sklearn_breast_cancer.csv")

print("\n" + "="*60)
print("DATASETS CREATED SUCCESSFULLY!")
print("="*60)
print("\nTest commands:")
print("\n1. DIGITS (with Neural Network):")
print("   python run.py --csv data/sklearn_digits.csv --target target")
print("\n2. WINE (traditional ML only):")
print("   python run.py --csv data/sklearn_wine.csv --target target")
print("\n3. BREAST CANCER (traditional ML only):")
print("   python run.py --csv data/sklearn_breast_cancer.csv --target target")
