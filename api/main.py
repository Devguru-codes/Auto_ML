"""
FastAPI application for AutoML
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import sys
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import load_model, load_json, create_artifacts_dir
from config.config import ARTIFACTS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME, METRICS_FILENAME

app = FastAPI(
    title="AutoTabML API",
    description="Automated Machine Learning for Tabular Data",
    version="1.0.0"
)

# Global variables to store loaded models
loaded_model = None
loaded_preprocessor = None
loaded_metrics = None


class PredictionInput(BaseModel):
    """Input schema for predictions"""
    data: List[Dict[str, Any]]


class TrainingResponse(BaseModel):
    """Response schema for training"""
    status: str
    message: str
    best_model: str
    metrics: Dict[str, Any]
    artifacts_path: str


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predictions: List[Any]
    model_used: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AutoTabML API",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train",
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = loaded_model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    metric: Optional[str] = Form(None)
):
    """
    Train AutoML model on uploaded CSV
    
    Args:
        file: CSV file
        target_column: Name of target column
        metric: Optional evaluation metric
        
    Returns:
        Training results and metrics
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate target column
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found. Available: {list(df.columns)}"
            )
        
        # Save uploaded file temporarily
        temp_csv_path = os.path.join(ARTIFACTS_DIR, "temp_upload.csv")
        create_artifacts_dir()
        df.to_csv(temp_csv_path, index=False)
        
        # Import and run training pipeline
        from run import run_automl
        
        results = run_automl(
            csv_path=temp_csv_path,
            target_column=target_column,
            metric=metric
        )
        
        # Load the trained model globally
        global loaded_model, loaded_preprocessor, loaded_metrics
        loaded_model = load_model(os.path.join(ARTIFACTS_DIR, MODEL_FILENAME))
        loaded_preprocessor = load_model(os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILENAME))
        loaded_metrics = load_json(os.path.join(ARTIFACTS_DIR, METRICS_FILENAME))
        
        return TrainingResponse(
            status="success",
            message="Model trained successfully",
            best_model=results['best_model_name'],
            metrics=results['best_metrics'],
            artifacts_path=ARTIFACTS_DIR
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Make predictions using trained model
    
    Args:
        input_data: JSON data for prediction
        
    Returns:
        Predictions
    """
    global loaded_model, loaded_preprocessor
    
    try:
        # Check if model is loaded
        if loaded_model is None or loaded_preprocessor is None:
            # Try to load from artifacts
            try:
                loaded_model = load_model(os.path.join(ARTIFACTS_DIR, MODEL_FILENAME))
                loaded_preprocessor = load_model(os.path.join(ARTIFACTS_DIR, PREPROCESSOR_FILENAME))
            except:
                raise HTTPException(
                    status_code=400,
                    detail="No trained model found. Please train a model first using /train endpoint"
                )
        
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Preprocess
        X_processed = loaded_preprocessor.transform(df)
        
        # Predict
        predictions = loaded_model.predict(X_processed)
        
        # Convert numpy types to Python types for JSON serialization
        predictions_list = [float(p) if isinstance(p, (np.floating, np.integer)) else p 
                          for p in predictions.tolist()]
        
        return PredictionResponse(
            predictions=predictions_list,
            model_used=loaded_model.__class__.__name__
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
