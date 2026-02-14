"""
Quick GPU Setup Script for AutoML
Automatically installs TensorFlow with GPU support
"""
import subprocess
import sys


def run_command(cmd):
    """Run a command and print output"""
    print(f"\n🔧 Running: {cmd}")
    print("=" * 80)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0


def main():
    print("=" * 80)
    print("🚀 AutoML GPU Setup")
    print("=" * 80)
    
    print("\n📋 This script will:")
    print("  1. Uninstall current TensorFlow")
    print("  2. Install TensorFlow with GPU support")
    print("  3. Verify GPU detection")
    
    response = input("\n⚠️  Continue? (y/n): ")
    if response.lower() != 'y':
        print("❌ Setup cancelled")
        return
    
    # Step 1: Uninstall TensorFlow
    print("\n" + "=" * 80)
    print("Step 1: Uninstalling current TensorFlow")
    print("=" * 80)
    run_command("pip uninstall tensorflow -y")
    
    # Step 2: Install GPU requirements
    print("\n" + "=" * 80)
    print("Step 2: Installing TensorFlow with GPU support")
    print("=" * 80)
    success = run_command("pip install -r requirements-gpu.txt")
    
    if not success:
        print("\n❌ Installation failed. Please check the error messages above.")
        print("\n💡 Try manual installation:")
        print("   pip install tensorflow[and-cuda]==2.15.0")
        return
    
    # Step 3: Verify GPU
    print("\n" + "=" * 80)
    print("Step 3: Verifying GPU detection")
    print("=" * 80)
    
    verify_code = """
import tensorflow as tf
print('\\n📊 TensorFlow GPU Status:')
print(f'  TensorFlow version: {tf.__version__}')
print(f'  Built with CUDA: {tf.test.is_built_with_cuda()}')
gpus = tf.config.list_physical_devices('GPU')
print(f'  GPU devices: {gpus}')
if gpus:
    print('\\n✅ GPU DETECTED! Your setup is complete.')
    print(f'  GPU Count: {len(gpus)}')
else:
    print('\\n⚠️  No GPU detected. See GPU_SETUP.md for troubleshooting.')
"""
    
    run_command(f'python -c "{verify_code}"')
    
    print("\n" + "=" * 80)
    print("✅ Setup Complete!")
    print("=" * 80)
    print("\n📝 Next steps:")
    print("  1. Run tests: python simple_tests.py")
    print("  2. Check GPU_SETUP.md if GPU not detected")
    print("  3. Start using AutoML with GPU acceleration!")


if __name__ == "__main__":
    main()
