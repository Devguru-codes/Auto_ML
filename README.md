# 🤖 AutoTabML - Enterprise-Grade AutoML Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoTabML** is a production-ready Automated Machine Learning platform that handles **Classification, Regression, and Time-Series Forecasting**. It features a modern web interface, REST API, and Docker deployment support.

![AutoTabML UI](https://via.placeholder.com/800x400?text=AutoTabML+Modern+Interface)

---

## ✨ Key Features

### 🧠 Core Capabilities
- **Multi-Modal Support**: Classification, Regression, and Time-Series Forecasting
- **Deep Learning Integration**: Auto-detects large datasets and trains **TensorFlow/Keras Neural Networks**
- **Smart Optimization**: Uses **Optuna (Bayesian Optimization)** for hyperparameter tuning
- **Advanced Time-Series**:
  - **Models**: ARIMA, Prophet, LSTM, GRU, Bidirectional LSTM
  - **Features**: Auto-seasonality detection, stationarity testing, sequence generation
- **Explainability**: SHAP values and Feature Importance charts

### 🚀 Production Ready
- **Modern Web UI**: Drag-and-drop interface, real-time progress, interactive charts
- **REST API**: Full-featured FastAPI backend with Swagger docs
- **Dockerized**: One-command deployment with Docker Compose
- **Visualizations**: Model comparison, Confusion Matrices, ROC Curves, Training History

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AutoML

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Locally (Web Interface)

```bash
# Start the server
python -m uvicorn api.main:app --reload --port 8000
```

Open your browser to: **http://localhost:8000**

### 3️⃣ Docker Deployment

```bash
docker-compose up -d
```

---

## 📊 Supported Models

| Problem Type | Traditional ML | Deep Learning |
|--------------|----------------|---------------|
| **Classification** | Logistic Regression, Random Forest, XGBoost, Gradient Boosting | Feed-Forward Neural Network (TensorFlow) |
| **Regression** | Linear Regression, Random Forest, XGBoost, Gradient Boosting | Feed-Forward Neural Network (TensorFlow) |
| **Time-Series** | ARIMA, Prophet | LSTM, GRU, Bidirectional LSTM |

**Note on Neural Networks:**
Neural networks are automatically enabled when the dataset size exceeds `1000` samples (configurable in `config/config.py`).

---

## 💻 CLI Usage

You can also run pipelines directly from the command line:

### Generate Test Data
```bash
python generate_data.py
```

### Classification/Regression
```bash
python run.py --csv data/classification_data.csv --target target
```

### Time-Series Forecasting
```bash
python run_timeseries.py \
  --csv data/stock_prices.csv \
  --target price \
  --date-column date \
  --sequence-length 20
```

---

## 📁 Project Structure

```
AutoML/
├── api/                 # FastAPI Backend
├── frontend/            # HTML/CSS/JS Frontend
├── models/              # Model Definitions (Sklearn + Keras)
├── timeseries/          # Time-Series Module (ARIMA, LSTM, etc.)
├── tuner/               # Optuna Hyperparameter Tuner
├── utils/               # Visualizations & Helpers
├── artifacts/           # Saved Models & Plots
├── Dockerfile           # Docker Config
├── docker-compose.yml   # Docker Compose
└── config/              # Configuration (Thresholds, Grids)
```

---

## 🔧 Configuration

Customize behavior in `config/config.py`:

- **NEURAL_NET_THRESHOLD**: Min samples to trigger NN training (Default: 1000)
- **OPTUNA_CONFIG**: Trials, timeout, and cross-validation settings
- **Model Hyperparameters**: Search spaces for all models

---

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request.

## 📄 License

MIT License. Built for the ML community.
