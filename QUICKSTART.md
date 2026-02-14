# 🚀 Quick Start Guide

## Run Locally (Easiest)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Open your browser:**
   - Web App: http://localhost:8000
   - API Docs: http://localhost:8000/api/docs

## Using the Web Interface

1. **Upload Data**: Drag & drop your CSV file
2. **Configure**: Select target column and problem type
3. **Train**: Click "Start Training" and watch progress
4. **View Results**: See metrics, visualizations, and model comparison

## Using the API

```bash
# Upload file
curl -X POST "http://localhost:8000/api/upload" -F "file=@data/iris.csv"

# Train model
curl -X POST "http://localhost:8000/api/train?file_id=<FILE_ID>" \
  -H "Content-Type: application/json" \
  -d '{"problem_type": "auto", "target_column": "species", "test_size": 0.2}'
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and production deployment.
