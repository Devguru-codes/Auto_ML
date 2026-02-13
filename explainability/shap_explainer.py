"""
SHAP-based explainability module
"""
import shap
import matplotlib.pyplot as plt
import numpy as np
import os


class ShapExplainer:
    """Generate SHAP explanations for model predictions"""
    
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self):
        """Create SHAP explainer based on model type"""
        print("Creating SHAP explainer...")
        
        # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
        model_name = self.model.__class__.__name__.lower()
        
        if any(tree_model in model_name for tree_model in ['forest', 'xgb', 'gradient', 'tree']):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # For linear models, use a sample of data for faster computation
            sample_size = min(100, len(self.X_train))
            background = shap.sample(self.X_train, sample_size)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        print("SHAP explainer created")
    
    def calculate_shap_values(self, X_test, max_samples=100):
        """Calculate SHAP values for test set"""
        if self.explainer is None:
            self.create_explainer()
        
        print("Calculating SHAP values...")
        
        # Limit samples for performance
        X_explain = X_test[:max_samples] if len(X_test) > max_samples else X_test
        
        try:
            self.shap_values = self.explainer.shap_values(X_explain)
            print(f"SHAP values calculated for {len(X_explain)} samples")
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            self.shap_values = None
        
        return self.shap_values
    
    def plot_summary(self, X_test, save_path=None, max_samples=100):
        """Generate and save SHAP summary plot"""
        if self.shap_values is None:
            self.calculate_shap_values(X_test, max_samples)
        
        if self.shap_values is None:
            print("Could not generate SHAP plot - no SHAP values available")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            # Limit samples
            X_plot = X_test[:max_samples] if len(X_test) > max_samples else X_test
            
            # Handle multiclass classification (use first class)
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            shap.summary_plot(shap_vals, X_plot, 
                            feature_names=self.feature_names,
                            show=False)
            
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error generating SHAP plot: {e}")
    
    def plot_feature_importance(self, X_test, save_path=None, max_samples=100):
        """Generate feature importance bar plot"""
        if self.shap_values is None:
            self.calculate_shap_values(X_test, max_samples)
        
        if self.shap_values is None:
            print("Could not generate feature importance - no SHAP values available")
            return
        
        try:
            # Handle multiclass
            shap_vals = self.shap_values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            # Calculate mean absolute SHAP values
            feature_importance = np.abs(shap_vals).mean(axis=0)
            
            # Create feature names if not provided
            if self.feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
            else:
                feature_names = self.feature_names
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1][:20]  # Top 20
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), feature_importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.gca().invert_yaxis()
            
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
