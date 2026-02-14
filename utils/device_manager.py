"""
Device Manager - GPU/CPU Detection and Configuration
"""
import os
import logging

logger = logging.getLogger(__name__)


def setup_tensorflow_device():
    """
    Configure TensorFlow to use GPU if available, otherwise CPU.
    
    Returns:
        str: Device type ('GPU' or 'CPU')
    """
    try:
        import tensorflow as tf
        
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to prevent TF from allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                logger.info(f"✓ GPU detected: {len(gpus)} device(s) available")
                logger.info(f"  GPU Names: {[gpu.name for gpu in gpus]}")
                return 'GPU'
            
            except RuntimeError as e:
                logger.warning(f"⚠ GPU detected but configuration failed: {e}")
                logger.info("  Falling back to CPU")
                return 'CPU'
        else:
            logger.info("ℹ No GPU detected. Using CPU for training.")
            return 'CPU'
    
    except ImportError:
        logger.warning("⚠ TensorFlow not installed. Skipping GPU detection.")
        return 'CPU'
    
    except Exception as e:
        logger.error(f"✗ Error during device setup: {e}")
        logger.info("  Falling back to CPU")
        return 'CPU'


def get_device_info():
    """
    Get detailed device information.
    
    Returns:
        dict: Device information including type, count, and names
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        
        return {
            'gpu_available': len(gpus) > 0,
            'gpu_count': len(gpus),
            'gpu_names': [gpu.name for gpu in gpus],
            'cpu_count': len(cpus),
            'device_type': 'GPU' if gpus else 'CPU'
        }
    
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        return {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'cpu_count': 1,
            'device_type': 'CPU'
        }


# Initialize device on module import
DEVICE_TYPE = setup_tensorflow_device()
