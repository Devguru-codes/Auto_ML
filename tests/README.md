# AutoML Test Suite

Comprehensive test suite for GPU support, error handling, and robustness features.

## 📋 Test Categories

### 1. GPU Detection Tests (`test_gpu_detection.py`)
Tests GPU/CUDA detection and CPU fallback mechanisms.

**What it tests:**
- ✅ Device setup returns valid type (GPU/CPU)
- ✅ Device info structure is correct
- ✅ CPU fallback works when GPU unavailable
- ✅ Multiple device setup calls are consistent
- ✅ TensorFlow integration
- ✅ GPU memory growth configuration

### 2. Error Handling Tests (`test_error_handling.py`)
Tests comprehensive error handling across the platform.

**What it tests:**
- ✅ Missing CSV file errors
- ✅ Invalid file path errors
- ✅ Missing target column errors
- ✅ Empty dataset handling
- ✅ Single-row dataset handling
- ✅ Non-time-series data detection
- ✅ Model training error handling
- ✅ API error handling

### 3. Model Training Tests (`test_model_training.py`)
Tests model creation and training with GPU support.

**What it tests:**
- ✅ NN Classifier creation
- ✅ NN Regressor creation
- ✅ NN Classifier training
- ✅ LSTM model creation
- ✅ GRU model creation
- ✅ Bidirectional LSTM creation
- ✅ LSTM training with GPU
- ✅ End-to-end classification pipeline

### 4. Logging Tests (`test_logging.py`)
Tests logging system configuration and output.

**What it tests:**
- ✅ Device manager logging
- ✅ Model zoo logging
- ✅ Time-series models logging
- ✅ API logging
- ✅ Log output verification

## 🚀 Running Tests

### Run All Tests
```bash
# From project root
pytest tests/ -v

# With detailed output
pytest tests/ -v -s

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test Suite
```bash
# GPU detection tests
pytest tests/test_gpu_detection.py -v -s

# Error handling tests
pytest tests/test_error_handling.py -v -s

# Model training tests
pytest tests/test_model_training.py -v -s

# Logging tests
pytest tests/test_logging.py -v -s
```

### Run Individual Test
```bash
# Example: Run only GPU detection test
pytest tests/test_gpu_detection.py::TestGPUDetection::test_device_setup_returns_valid_type -v
```

## 📊 Expected Results

### On GPU-Enabled System
```
✓ GPU detected: 1 device(s) available
✓ Device detected: GPU
✓ GPU memory growth configured
✓ All models use GPU for training
```

### On CPU-Only System
```
ℹ No GPU detected. Using CPU for training.
✓ Device detected: CPU
✓ CPU fallback working correctly
✓ All models train on CPU
```

## 🔍 Pre-PR Checklist

Before creating a PR, ensure all tests pass:

1. **Install test dependencies:**
   ```bash
   pip install pytest pytest-cov
   ```

2. **Run full test suite:**
   ```bash
   pytest tests/ -v
   ```

3. **Check for warnings:**
   ```bash
   pytest tests/ -v -W error
   ```

4. **Verify GPU detection (if GPU available):**
   ```bash
   python -c "from utils.device_manager import get_device_info; print(get_device_info())"
   ```

5. **Test error scenarios:**
   ```bash
   pytest tests/test_error_handling.py -v -s
   ```

## 🐛 Troubleshooting

### TensorFlow Import Errors
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

### CUDA Not Found (Windows)
```bash
# Check CUDA installation
nvidia-smi

# Verify TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Test Failures
1. Check log output in test results
2. Verify all dependencies are installed
3. Ensure you're in the project root directory
4. Check Python version (3.10+ required)

## 📝 Adding New Tests

When adding new features, create corresponding tests:

1. Create test file in `tests/` directory
2. Follow naming convention: `test_<feature_name>.py`
3. Use pytest fixtures for common setup
4. Add docstrings explaining what each test validates
5. Update this README with new test category

## 🎯 Coverage Goals

- **GPU Detection**: 100% coverage
- **Error Handling**: 90%+ coverage
- **Model Training**: 85%+ coverage
- **Logging**: 100% coverage

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [TensorFlow Testing Guide](https://www.tensorflow.org/guide/test)
- [Python unittest](https://docs.python.org/3/library/unittest.html)
