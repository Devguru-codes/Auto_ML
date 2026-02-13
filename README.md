# 🤖 AutoTabML - Automated Machine Learning for Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoTabML** is a fully automated Machine Learning pipeline for tabular datasets. Upload a CSV, specify the target column, and get a trained, optimized model with explanations—all automatically!

## ✨ Features

- 🎯 **Automatic Problem Detection** - Detects classification vs regression
- 🔧 **Smart Preprocessing** - Handles numeric & categorical features automatically
- 🏆 **Model Zoo** - Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- ⚡ **Hyperparameter Tuning** - Uses RandomizedSearchCV for optimization
- 📊 **Model Evaluation** - Comprehensive metrics (Accuracy, F1, ROC-AUC, RMSE, MAE, R²)
- 🔍 **Explainability** - SHAP-based feature importance and explanations
- 🚀 **REST API** - FastAPI endpoints for training and prediction
- 💾 **Model Persistence** - Saves best model and preprocessing pipeline

---

## 📁 Project Structure

```
AutoML/
│
├── config/
│   └── config.py              # Configuration settings
├── data/
│   ├── classification_example.csv
│   └── regression_example.csv
├── preprocessing/
│   └── preprocessor.py        # Automatic preprocessing pipeline
├── models/
│   └── model_zoo.py           # Model factory
├── tuner/
│   └── tuner.py               # Hyperparameter tuning
├── evaluator/
│   └── evaluator.py           # Model evaluation
├── explainability/
│   └── shap_explainer.py      # SHAP explanations
├── api/
│   └── main.py                # FastAPI application
├── utils/
│   ├── utils.py               # Utility functions
│   └── problem_detector.py    # Problem type detection
├── artifacts/                 # Saved models and plots
├── run.py                     # Main pipeline runner
├── generate_example_data.py   # Generate test datasets
└── requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AutoML

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Generate Example Data

```bash
python generate_example_data.py
```

This creates two example datasets:
- `data/classification_example.csv` - Multi-class classification
- `data/regression_example.csv` - House price prediction

### 3️⃣ Run AutoML Pipeline

**Classification Example:**
```bash
python run.py --csv data/classification_example.csv --target target
```

**Regression Example:**
```bash
python run.py --csv data/regression_example.csv --target price
```

**With Custom Metric:**
```bash
python run.py --csv data/classification_example.csv --target target --metric f1_weighted
```

### 4️⃣ Start API Server

```bash
# From api directory
cd api
uvicorn main:app --reload

# Or directly
python api/main.py
```

API will be available at: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## 📡 API Usage

### Training Endpoint

**POST** `/train`

Upload a CSV file and train a model:

```bash
curl -X POST "http://localhost:8000/train" \
  -F "file=@data/classification_example.csv" \
  -F "target_column=target"
```

**Response:**
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "best_model": "xgboost",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.94,
    "roc_auc": 0.98
  },
  "artifacts_path": "artifacts"
}
```

### Prediction Endpoint

**POST** `/predict`

Make predictions with trained model:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "feature_0": 1.5,
        "feature_1": -0.3,
        "category_A": "Type1",
        "category_B": "High"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [2],
  "model_used": "XGBClassifier"
}
```

---

## 🎯 How It Works

### 1. **Problem Detection**
- Analyzes target variable
- If numeric with >20 unique values → Regression
- Otherwise → Classification

### 2. **Preprocessing**
- **Numeric features**: Mean imputation → StandardScaler
- **Categorical features**: Mode imputation → OneHotEncoder
- Uses sklearn `ColumnTransformer` and `Pipeline`

### 3. **Model Training**
Trains 4 models for each problem type:

**Classification:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

**Regression:**
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### 4. **Hyperparameter Tuning**
- Uses `RandomizedSearchCV`
- 5-fold cross-validation
- 20 iterations per model

### 5. **Model Evaluation**

**Classification Metrics:**
- Accuracy
- F1-Score (weighted)
- ROC-AUC

**Regression Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

### 6. **Explainability**
- Generates SHAP summary plots
- Feature importance visualization
- Saved to `artifacts/` directory

---

## 📊 Output Artifacts

After training, the following files are saved in `artifacts/`:

| File | Description |
|------|-------------|
| `best_model.pkl` | Trained best model |
| `preprocessor.pkl` | Fitted preprocessing pipeline |
| `metrics.json` | All model metrics |
| `shap_summary.png` | SHAP summary plot |
| `feature_importance.png` | Feature importance chart |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn |
| Gradient Boosting | XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV |
| Explainability | SHAP |
| API | FastAPI |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📝 Command Line Arguments

```bash
python run.py --help
```

**Arguments:**
- `--csv` (required): Path to CSV file
- `--target` (required): Target column name
- `--metric` (optional): Evaluation metric (auto-selected if not provided)
- `--test-size` (optional): Test set fraction (default: 0.2)

---

## 🎓 Example Workflow

```python
# 1. Generate example data
python generate_example_data.py

# 2. Train model
python run.py --csv data/classification_example.csv --target target

# 3. Check artifacts
ls artifacts/
# Output: best_model.pkl, preprocessor.pkl, metrics.json, shap_summary.png, feature_importance.png

# 4. Start API
python api/main.py

# 5. Make predictions via API
# (Use curl or Postman as shown above)
```

---

## 🔮 Future Enhancements (V2)

- [ ] LightGBM and CatBoost support
- [ ] Bayesian Optimization (Optuna)
- [ ] Auto feature engineering
- [ ] Model ensembling (Stacking)
- [ ] Streamlit UI
- [ ] Docker deployment
- [ ] Time-series support
- [ ] MLflow integration

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Built with ❤️ for the ML community

---

## 🙏 Acknowledgments

- scikit-learn team
- XGBoost developers
- SHAP library creators
- FastAPI framework

---

**Happy AutoML-ing! 🚀**
