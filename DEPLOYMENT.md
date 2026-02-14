# 🚀 AutoTabML Deployment Guide

## Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Application
Open your browser and navigate to:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Alternative Docs**: http://localhost:8000/api/redoc

---

## Docker Deployment

### Build and Run with Docker Compose
```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t automl:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/artifacts:/app/artifacts \
  automl:latest
```

---

## Production Deployment

### Environment Variables
Create a `.env` file:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Storage
UPLOAD_DIR=uploads
MODELS_DIR=saved_models
ARTIFACTS_DIR=artifacts

# CORS (comma-separated origins)
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Increase timeouts for long-running training
    proxy_read_timeout 600s;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
}
```

### Systemd Service
Create `/etc/systemd/system/automl.service`:
```ini
[Unit]
Description=AutoTabML Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/AutoML
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable automl
sudo systemctl start automl
sudo systemctl status automl
```

---

## Cloud Deployment

### AWS EC2
1. Launch Ubuntu instance (t3.medium or larger)
2. Install Docker:
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker $USER
   ```
3. Clone repository and run:
   ```bash
   git clone <your-repo>
   cd AutoML
   docker-compose up -d
   ```

### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/automl

# Deploy
gcloud run deploy automl \
  --image gcr.io/PROJECT_ID/automl \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 600
```

### Heroku
```bash
# Login
heroku login

# Create app
heroku create your-automl-app

# Set stack to container
heroku stack:set container

# Deploy
git push heroku main
```

---

## API Usage Examples

### Upload File
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@data/iris.csv"
```

### Train Model
```bash
curl -X POST "http://localhost:8000/api/train?file_id=<FILE_ID>" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "classification",
    "target_column": "species",
    "test_size": 0.2
  }'
```

### Check Job Status
```bash
curl "http://localhost:8000/api/jobs/<JOB_ID>"
```

### List Models
```bash
curl "http://localhost:8000/api/models"
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

### Memory Issues
- Increase Docker memory limit in Docker Desktop settings
- Use smaller datasets for testing
- Reduce number of Optuna trials in `config/config.py`

### CORS Errors
Update `api/main.py` to specify exact origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    ...
)
```

---

## Performance Tuning

### For Large Datasets
- Increase worker count: `--workers 4`
- Use Redis for job queue (replace in-memory dict)
- Enable caching for preprocessing

### For Production
- Use Gunicorn with Uvicorn workers:
  ```bash
  gunicorn api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
  ```

---

## Monitoring

### Health Check
```bash
curl http://localhost:8000/
```

### Logs
```bash
# Docker
docker-compose logs -f

# Systemd
sudo journalctl -u automl -f
```

---

## Security Recommendations

1. **Use HTTPS** in production (Let's Encrypt)
2. **Add authentication** for sensitive deployments
3. **Limit file upload size** in nginx/API
4. **Sanitize file uploads** (already done for CSV)
5. **Use environment variables** for secrets
6. **Enable rate limiting** (e.g., with nginx)

---

## Support

For issues or questions:
- Check API docs: http://localhost:8000/api/docs
- Review logs for errors
- Ensure all dependencies are installed
