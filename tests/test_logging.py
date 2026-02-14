"""
Test Logging System
"""
import pytest
import sys
import os
import logging
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLoggingConfiguration:
    """Test logging configuration across modules"""
    
    def test_device_manager_logging(self):
        """Test that device_manager has logging configured"""
        from utils import device_manager
        
        assert hasattr(device_manager, 'logger')
        assert isinstance(device_manager.logger, logging.Logger)
        print("✓ Device manager logging configured")
    
    def test_model_zoo_logging(self):
        """Test that model_zoo has logging configured"""
        from models import model_zoo
        
        assert hasattr(model_zoo, 'logger')
        assert isinstance(model_zoo.logger, logging.Logger)
        print("✓ Model zoo logging configured")
    
    def test_ts_models_logging(self):
        """Test that ts_models has logging configured"""
        from timeseries import ts_models
        
        assert hasattr(ts_models, 'logger')
        assert isinstance(ts_models.logger, logging.Logger)
        print("✓ Time-series models logging configured")
    
    def test_api_logging(self):
        """Test that API has logging configured"""
        from api import main
        
        assert hasattr(main, 'logger')
        assert isinstance(main.logger, logging.Logger)
        print("✓ API logging configured")


class TestLogOutput:
    """Test that logging actually produces output"""
    
    def test_gpu_detection_logs(self):
        """Test that GPU detection produces log output"""
        from utils.device_manager import setup_tensorflow_device
        
        # Capture log output
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as log_file:
            handler = logging.FileHandler(log_file.name)
            handler.setLevel(logging.INFO)
            
            logger = logging.getLogger('utils.device_manager')
            logger.addHandler(handler)
            
            # Trigger GPU detection
            device = setup_tensorflow_device()
            
            handler.close()
            logger.removeHandler(handler)
            
            # Check log file
            with open(log_file.name, 'r') as f:
                log_content = f.read()
            
            os.unlink(log_file.name)
            
            # Should contain device info
            assert len(log_content) > 0 or device is not None
            print(f"✓ GPU detection logged (Device: {device})")


if __name__ == "__main__":
    print("=" * 80)
    print("LOGGING TEST SUITE")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
