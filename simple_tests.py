"""
Simple Integration Tests for GPU Support and Error Handling
These tests validate core functionality without complex dependencies
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gpu_detection_module_exists():
    """Test that GPU detection module can be imported"""
    try:
        from utils import device_manager
        print("✅ GPU detection module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import device_manager: {e}")
        return False


def test_device_setup_function():
    """Test that device setup function works"""
    try:
        from utils.device_manager import setup_tensorflow_device
        device = setup_tensorflow_device()
        assert device in ['GPU', 'CPU'], f"Invalid device type: {device}"
        print(f"✅ Device setup successful: {device}")
        return True
    except Exception as e:
        print(f"❌ Device setup failed: {e}")
        return False


def test_device_info_function():
    """Test that device info function returns valid data"""
    try:
        from utils.device_manager import get_device_info
        info = get_device_info()
        
        # Check for actual keys returned by the function
        required_keys = ['device_type', 'gpu_available', 'gpu_count', 'gpu_names', 'cpu_count']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        print(f"✅ Device info retrieved successfully")
        print(f"   Device Type: {info['device_type']}")
        print(f"   GPU Available: {info['gpu_available']}")
        print(f"   GPU Count: {info['gpu_count']}")
        print(f"   CPU Count: {info['cpu_count']}")
        return True
    except Exception as e:
        print(f"❌ Device info failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_in_run():
    """Test that run.py has error handling"""
    try:
        with open('run.py', 'r') as f:
            content = f.read()
        
        assert 'try:' in content, "No try block found"
        assert 'except' in content, "No except block found"
        assert 'logger' in content, "No logger found"
        
        print("✅ Error handling present in run.py")
        return True
    except Exception as e:
        print(f"❌ Error handling check failed: {e}")
        return False


def test_error_handling_in_run_timeseries():
    """Test that run_timeseries.py has error handling"""
    try:
        with open('run_timeseries.py', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        assert 'try:' in content, "No try block found"
        assert 'except' in content, "No except block found"
        assert 'logger' in content, "No logger found"
        
        print("✅ Error handling present in run_timeseries.py")
        return True
    except Exception as e:
        print(f"❌ Error handling check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_error_handling():
    """Test that API has error handling"""
    try:
        with open('api/main.py', 'r') as f:
            content = f.read()
        
        assert 'logger' in content, "No logger found"
        assert 'exception_handler' in content, "No exception handler found"
        
        print("✅ Error handling present in API")
        return True
    except Exception as e:
        print(f"❌ API error handling check failed: {e}")
        return False


def test_logging_configuration():
    """Test that logging is configured in modules"""
    try:
        modules_to_check = [
            ('utils/device_manager.py', 'device_manager'),
            ('models/model_zoo.py', 'model_zoo'),
            ('timeseries/ts_models.py', 'ts_models'),
            ('api/main.py', 'API')
        ]
        
        all_passed = True
        for file_path, name in modules_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if 'logging' in content and 'logger' in content:
                print(f"✅ Logging configured in {name}")
            else:
                print(f"❌ Logging missing in {name}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Logging check failed: {e}")
        return False


def test_tensorflow_import():
    """Test that TensorFlow can be imported"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully (version {tf.__version__})")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False


def test_model_zoo_has_gpu_support():
    """Test that model_zoo.py has GPU support code"""
    try:
        with open('models/model_zoo.py', 'r') as f:
            content = f.read()
        
        assert 'setup_tensorflow_device' in content, "No GPU detection call found"
        
        print("✅ Model zoo has GPU support")
        return True
    except Exception as e:
        print(f"❌ Model zoo GPU check failed: {e}")
        return False


def test_ts_models_have_gpu_support():
    """Test that ts_models.py has GPU support code"""
    try:
        with open('timeseries/ts_models.py', 'r') as f:
            content = f.read()
        
        assert 'setup_tensorflow_device' in content, "No GPU detection call found"
        
        print("✅ Time-series models have GPU support")
        return True
    except Exception as e:
        print(f"❌ Time-series models GPU check failed: {e}")
        return False


def main():
    """Run all simple integration tests"""
    print("=" * 80)
    print("🧪 SIMPLE INTEGRATION TESTS - GPU Support & Error Handling")
    print("=" * 80)
    print()
    
    tests = [
        ("GPU Detection Module", test_gpu_detection_module_exists),
        ("Device Setup Function", test_device_setup_function),
        ("Device Info Function", test_device_info_function),
        ("Error Handling in run.py", test_error_handling_in_run),
        ("Error Handling in run_timeseries.py", test_error_handling_in_run_timeseries),
        ("API Error Handling", test_api_error_handling),
        ("Logging Configuration", test_logging_configuration),
        ("TensorFlow Import", test_tensorflow_import),
        ("Model Zoo GPU Support", test_model_zoo_has_gpu_support),
        ("Time-Series Models GPU Support", test_ts_models_have_gpu_support),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        print("-" * 80)
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for PR!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Please review")
        return 1


if __name__ == "__main__":
    exit(main())
