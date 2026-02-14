"""
Test Model Training with GPU Support
"""
import pytest
import sys
import os
import tempfile
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_zoo import create_nn_classifier, create_nn_regressor
from timeseries.ts_models import LSTMModel, GRUModel, BidirectionalLSTMModel


class TestNeuralNetworkModels:
    """Test neural network models with GPU support"""
    
    def test_nn_classifier_creation(self):
        """Test that NN classifier can be created"""
        try:
            model = create_nn_classifier(
                hidden_layer_sizes=(32, 16),
                learning_rate=0.001,
                input_dim=10
            )
            assert model is not None
            print("✓ NN Classifier created successfully")
        except Exception as e:
            pytest.fail(f"NN Classifier creation failed: {e}")
    
    def test_nn_regressor_creation(self):
        """Test that NN regressor can be created"""
        try:
            model = create_nn_regressor(
                hidden_layer_sizes=(32, 16),
                learning_rate=0.001,
                input_dim=10
            )
            assert model is not None
            print("✓ NN Regressor created successfully")
        except Exception as e:
            pytest.fail(f"NN Regressor creation failed: {e}")
    
    def test_nn_classifier_training(self):
        """Test that NN classifier can be trained"""
        try:
            from scikeras.wrappers import KerasClassifier
            
            # Create synthetic data
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            # Create and train model
            model = create_nn_classifier(input_dim=10)
            keras_clf = KerasClassifier(model=model, epochs=2, batch_size=16, verbose=0)
            keras_clf.fit(X, y)
            
            # Make predictions
            predictions = keras_clf.predict(X[:5])
            assert len(predictions) == 5
            print("✓ NN Classifier training successful")
        except Exception as e:
            pytest.fail(f"NN Classifier training failed: {e}")


class TestTimeSeriesModels:
    """Test time-series deep learning models with GPU support"""
    
    def test_lstm_model_creation(self):
        """Test that LSTM model can be created"""
        try:
            model = LSTMModel(units=32, layers=1, dropout=0.2)
            assert model is not None
            print("✓ LSTM Model created successfully")
        except Exception as e:
            pytest.fail(f"LSTM Model creation failed: {e}")
    
    def test_gru_model_creation(self):
        """Test that GRU model can be created"""
        try:
            model = GRUModel(units=32, layers=1, dropout=0.2)
            assert model is not None
            print("✓ GRU Model created successfully")
        except Exception as e:
            pytest.fail(f"GRU Model creation failed: {e}")
    
    def test_bidirectional_lstm_creation(self):
        """Test that Bidirectional LSTM can be created"""
        try:
            model = BidirectionalLSTMModel(units=32, layers=1, dropout=0.2)
            assert model is not None
            print("✓ Bidirectional LSTM created successfully")
        except Exception as e:
            pytest.fail(f"Bidirectional LSTM creation failed: {e}")
    
    def test_lstm_training(self):
        """Test that LSTM can be trained"""
        try:
            # Create synthetic time-series data
            X_train = np.random.rand(50, 10, 1)  # (samples, timesteps, features)
            y_train = np.random.rand(50, 1)
            X_val = np.random.rand(10, 10, 1)
            y_val = np.random.rand(10, 1)
            
            model = LSTMModel(units=16, layers=1, dropout=0.2, epochs=2)
            model.fit(X_train, y_train, X_val, y_val)
            
            # Make predictions
            predictions = model.predict(X_val)
            assert predictions.shape[0] == 10
            print("✓ LSTM training successful")
        except Exception as e:
            pytest.fail(f"LSTM training failed: {e}")


class TestEndToEndPipeline:
    """Test end-to-end pipeline with small dataset"""
    
    def test_classification_pipeline(self):
        """Test complete classification pipeline"""
        from run import run_automl
        
        # Create small test dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("feature1,feature2,feature3,target\n")
            for i in range(50):
                f.write(f"{np.random.rand()},{np.random.rand()},{np.random.rand()},{i%2}\n")
            temp_path = f.name
        
        try:
            result = run_automl(
                csv_path=temp_path,
                target_column="target",
                test_size=0.3
            )
            
            assert 'best_model_name' in result
            assert 'best_metrics' in result
            print(f"✓ Classification pipeline successful. Best model: {result['best_model_name']}")
        except Exception as e:
            pytest.fail(f"Classification pipeline failed: {e}")
        finally:
            os.unlink(temp_path)
            # Clean up artifacts
            if 'artifacts_path' in result:
                import shutil
                if os.path.exists(result['artifacts_path']):
                    shutil.rmtree(result['artifacts_path'])


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL TRAINING TEST SUITE")
    print("=" * 80)
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
