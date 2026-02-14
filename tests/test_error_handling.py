"""
Test Error Handling and Robustness
"""
import pytest
import sys
import os
import pandas as pd
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import run_automl
from run_timeseries import run_timeseries_automl


class TestFileErrorHandling:
    """Test error handling for file operations"""
    
    def test_missing_csv_file(self):
        """Test that missing CSV file raises appropriate error"""
        with pytest.raises(FileNotFoundError):
            run_automl(
                csv_path="nonexistent_file.csv",
                target_column="target"
            )
        print("✓ Missing file error handled correctly")
    
    def test_invalid_file_path(self):
        """Test that invalid file path raises error"""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            run_automl(
                csv_path="/invalid/path/to/file.csv",
                target_column="target"
            )
        print("✓ Invalid path error handled correctly")


class TestDataValidationErrors:
    """Test error handling for data validation"""
    
    def test_missing_target_column(self):
        """Test that missing target column raises KeyError"""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2\n")
            f.write("1,2\n")
            f.write("3,4\n")
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                run_automl(
                    csv_path=temp_path,
                    target_column="nonexistent_target"
                )
            print("✓ Missing target column error handled correctly")
        finally:
            os.unlink(temp_path)
    
    def test_empty_dataset(self):
        """Test that empty dataset is handled"""
        # Create empty CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,target\n")
            temp_path = f.name
        
        try:
            with pytest.raises((ValueError, RuntimeError)):
                run_automl(
                    csv_path=temp_path,
                    target_column="target"
                )
            print("✓ Empty dataset error handled correctly")
        finally:
            os.unlink(temp_path)
    
    def test_single_row_dataset(self):
        """Test that single-row dataset is handled"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,target\n")
            f.write("1,2,0\n")
            temp_path = f.name
        
        try:
            with pytest.raises((ValueError, RuntimeError)):
                run_automl(
                    csv_path=temp_path,
                    target_column="target"
                )
            print("✓ Single-row dataset error handled correctly")
        finally:
            os.unlink(temp_path)


class TestTimeSeriesErrorHandling:
    """Test error handling for time-series pipeline"""
    
    def test_non_timeseries_data(self):
        """Test that non-time-series data raises ValueError"""
        # Create non-time-series CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,target\n")
            f.write("1,2,10\n")
            f.write("3,4,20\n")
            f.write("5,6,30\n")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                run_timeseries_automl(
                    csv_path=temp_path,
                    target_column="target"
                )
            assert "not time-series" in str(exc_info.value).lower()
            print("✓ Non-time-series data error handled correctly")
        finally:
            os.unlink(temp_path)


class TestModelTrainingErrors:
    """Test error handling during model training"""
    
    def test_invalid_problem_type(self):
        """Test that invalid problem type is handled"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,target\n")
            for i in range(20):
                f.write(f"{i},{i*2},{i%2}\n")
            temp_path = f.name
        
        try:
            # This should work or raise a clear error
            try:
                result = run_automl(
                    csv_path=temp_path,
                    target_column="target",
                    test_size=0.3
                )
                print("✓ Model training completed successfully")
            except Exception as e:
                # Should be a clear, logged error
                assert isinstance(e, (ValueError, RuntimeError))
                print(f"✓ Model training error handled: {type(e).__name__}")
        finally:
            os.unlink(temp_path)


class TestAPIErrorHandling:
    """Test API error handling (requires API to be running)"""
    
    def test_api_imports(self):
        """Test that API modules can be imported without errors"""
        try:
            from api.main import app
            assert app is not None
            print("✓ API imports successful")
        except Exception as e:
            pytest.fail(f"API import failed: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("ERROR HANDLING TEST SUITE")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
