"""
Enhanced FastAPI backend for AutoML with full deployment support
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import pandas as pd
import json
import uuid
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="AutoTabML API",
    description="Automated Machine Learning API with Classification, Regression, and Time-Series support",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
    )

# Create necessary directories
UPLOAD_DIR = "uploads"
MODELS_DIR = "saved_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory job storage
jobs = {}


class TrainRequest(BaseModel):
    problem_type: str
    target_column: str
    date_column: Optional[str] = None
    test_size: float = 0.2
    sequence_length: int = 10


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AutoTabML API v2.0",
        "status": "online",
        "endpoints": {
            "docs": "/api/docs",
            "upload": "/api/upload",
            "train": "/api/train",
            "jobs": "/api/jobs/{job_id}",
            "models": "/api/models"
        }
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file and return preview"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files supported")
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        df = pd.read_csv(file_path)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict('records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_model(file_id: str, request: TrainRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "file_id": file_id,
            "request": request.dict()
        }
        
        background_tasks.add_task(train_model_task, job_id, file_path, request)
        
        return {"job_id": job_id, "status": "queued"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def train_model_task(job_id: str, file_path: str, request: TrainRequest):
    """Background training task"""
    try:
        logger.info(f"Starting training job {job_id} for file {file_path}")
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 10
        
        if request.problem_type == 'timeseries':
            from run_timeseries import run_timeseries_automl
            result = run_timeseries_automl(
                csv_path=file_path,
                target_column=request.target_column,
                date_column=request.date_column,
                test_size=request.test_size,
                sequence_length=request.sequence_length
            )
        else:
            from run import run_automl
            result = run_automl(
                csv_path=file_path,
                target_column=request.target_column,
                test_size=request.test_size,
                problem_type=None if request.problem_type == 'auto' else request.problem_type
            )
        
        jobs[job_id]["progress"] = 90
        
        model_id = str(uuid.uuid4())
        model_info = {
            "model_id": model_id,
            "job_id": job_id,
            "problem_type": request.problem_type,
            "best_model": result.get('best_model_name'),
            "metrics": result.get('best_metrics', {}),
            "artifacts_path": result.get('artifacts_path'),
            "created_at": datetime.now().isoformat()
        }
        
        model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
        with open(model_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["model_id"] = model_id
        jobs[job_id]["result"] = model_info
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/models")
async def list_models():
    """List all models"""
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(MODELS_DIR, filename), 'r') as f:
                models.append(json.load(f))
    return {"models": sorted(models, key=lambda x: x['created_at'], reverse=True)}


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    with open(model_path, 'r') as f:
        return json.load(f)


@app.get("/api/artifacts/{model_id}/{filename}")
async def get_artifact(model_id: str, filename: str):
    """Get model artifact"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    with open(model_path, 'r') as f:
        model_info = json.load(f)
    
    artifact_path = os.path.join(model_info['artifacts_path'], filename)
    if not os.path.exists(artifact_path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return FileResponse(artifact_path)


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete model"""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    os.remove(model_path)
    return {"message": "Model deleted"}


# Mount frontend
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
