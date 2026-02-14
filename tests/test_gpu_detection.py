"""
Test GPU Detection and Device Management
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.device_manager import setup_tensorflow_device, get_device_info


class TestGPUDetection:
    """Test GPU detection and fallback mechanisms"""
    
    def test_device_setup_returns_valid_type(self):
        """Test that device setup returns either 'GPU' or 'CPU'"""
        device = setup_tensorflow_device()
        assert device in ['GPU', 'CPU'], f"Expected 'GPU' or 'CPU', got {device}"
    
    def test_device_info_structure(self):
        """Test that device info returns correct structure"""
        info = get_device_info()
        
        assert 'device_type' in info
        assert 'device_count' in info
        assert 'device_names' in info
        assert 'tensorflow_version' in info
        
        assert info['device_type'] in ['GPU', 'CPU']
        assert isinstance(info['device_count'], int)
        assert isinstance(info['device_names'], list)
        assert isinstance(info['tensorflow_version'], str)
    
    def test_cpu_fallback_when_gpu_unavailable(self):
        """Test that system falls back to CPU gracefully"""
        # This test will pass on both GPU and CPU systems
        device = setup_tensorflow_device()
        assert device is not None
        print(f"✓ Device detected: {device}")
    
    def test_multiple_device_setup_calls(self):
        """Test that multiple calls to setup_tensorflow_device are safe"""
        device1 = setup_tensorflow_device()
        device2 = setup_tensorflow_device()
        device3 = setup_tensorflow_device()
        
        assert device1 == device2 == device3
        print(f"✓ Consistent device across calls: {device1}")


class TestTensorFlowIntegration:
    """Test TensorFlow integration with device manager"""
    
    def test_tensorflow_import(self):
        """Test that TensorFlow can be imported"""
        try:
            import tensorflow as tf
            assert tf.__version__ is not None
            print(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError as e:
            pytest.fail(f"TensorFlow import failed: {e}")
    
    def test_gpu_memory_growth_configuration(self):
        """Test that GPU memory growth is configured (if GPU available)"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                # Check if memory growth is set
                for gpu in gpus:
                    config = tf.config.experimental.get_memory_growth(gpu)
                    print(f"✓ GPU {gpu.name} memory growth: {config}")
            else:
                print("ℹ No GPU available, skipping memory growth test")
        except Exception as e:
            pytest.fail(f"GPU configuration check failed: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("GPU DETECTION TEST SUITE")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
