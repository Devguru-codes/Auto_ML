"""
Quick Test Runner for AutoML
Run this before creating a PR to validate all changes
"""
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print results"""
    print("\n" + "=" * 80)
    print(f"🧪 {description}")
    print("=" * 80)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ {description} FAILED")
        return False
    else:
        print(f"✅ {description} PASSED")
        return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 AutoML Pre-PR Test Suite")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: GPU Detection
    all_passed &= run_command(
        "pytest tests/test_gpu_detection.py -v -s",
        "GPU Detection Tests"
    )
    
    # Test 2: Error Handling
    all_passed &= run_command(
        "pytest tests/test_error_handling.py -v -s",
        "Error Handling Tests"
    )
    
    # Test 3: Model Training
    all_passed &= run_command(
        "pytest tests/test_model_training.py -v -s",
        "Model Training Tests"
    )
    
    # Test 4: Logging
    all_passed &= run_command(
        "pytest tests/test_logging.py -v -s",
        "Logging Tests"
    )
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for PR!")
    else:
        print("❌ SOME TESTS FAILED - Please fix before PR")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
