"""
Visualization module for model performance and metrics
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import os


class ModelVisualizer:
    """Generate comprehensive visualizations for model performance"""
    
    def __init__(self, artifacts_dir='artifacts'):
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_model_comparison(self, results_df, problem_type, save_path=None):
        """
        Create bar chart comparing all models
        
        Args:
            results_df: DataFrame with model results
            problem_type: 'classification' or 'regression'
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3 if problem_type == 'classification' else 3, figsize=(15, 5))
        
        if problem_type == 'classification':
            metrics = ['accuracy', 'f1_score', 'roc_auc']
            titles = ['Accuracy Comparison', 'F1-Score Comparison', 'ROC-AUC Comparison']
        else:
            metrics = ['r2', 'rmse', 'mae']
            titles = ['R² Score Comparison', 'RMSE Comparison', 'MAE Comparison']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in results_df.columns:
                ax = axes[idx] if len(metrics) > 1 else axes
                
                # Sort by metric (ascending for error metrics, descending for scores)
                ascending = metric in ['rmse', 'mae']
                sorted_df = results_df.sort_values(metric, ascending=ascending)
                
                # Create bar plot
                bars = ax.barh(sorted_df['model_name'], sorted_df[metric])
                
                # Color bars - best model in green
                colors = ['#2ecc71' if i == (0 if ascending else len(bars)-1) else '#3498db' 
                         for i in range(len(bars))]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_xlabel(metric.upper().replace('_', ' '))
                ax.set_title(title, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (idx_val, row) in enumerate(sorted_df.iterrows()):
                    ax.text(row[metric], i, f' {row[metric]:.4f}', 
                           va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy per class
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            plt.text(len(cm) + 0.5, i + 0.5, f'{acc:.2%}', 
                    ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(self, models_dict, X_test, y_test, problem_type, save_path=None):
        """
        Plot ROC curves for all models
        
        Args:
            models_dict: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            problem_type: 'classification' or 'regression'
            save_path: Path to save the plot
        """
        if problem_type != 'classification':
            print("ROC curves only available for classification")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Get unique classes
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_dict)))
        
        for (model_name, model), color in zip(models_dict.items(), colors):
            try:
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_test)
                    
                    if n_classes == 2:
                        # Binary classification
                        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, color=color, lw=2,
                                label=f'{model_name} (AUC = {roc_auc:.3f})')
                    else:
                        # Multi-class: compute macro-average
                        y_test_bin = label_binarize(y_test, classes=classes)
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        # Compute macro-average
                        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                        mean_tpr = np.zeros_like(all_fpr)
                        for i in range(n_classes):
                            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                        mean_tpr /= n_classes
                        
                        mean_auc = auc(all_fpr, mean_tpr)
                        plt.plot(all_fpr, mean_tpr, color=color, lw=2,
                                label=f'{model_name} (macro-avg AUC = {mean_auc:.3f})')
            except Exception as e:
                print(f"Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training and validation curves for neural networks
        
        Args:
            history: Keras history object
            save_path: Path to save the plot
        """
        if history is None or not hasattr(history, 'history'):
            print("No training history available")
            return
        
        history_dict = history.history
        
        # Determine metrics to plot
        metrics_to_plot = []
        if 'loss' in history_dict:
            metrics_to_plot.append(('loss', 'Loss'))
        if 'accuracy' in history_dict:
            metrics_to_plot.append(('accuracy', 'Accuracy'))
        if 'mae' in history_dict:
            metrics_to_plot.append(('mae', 'MAE'))
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            print("No metrics found in history")
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Plot training metric
            epochs = range(1, len(history_dict[metric_key]) + 1)
            ax.plot(epochs, history_dict[metric_key], 'b-', label=f'Training {metric_name}', linewidth=2)
            
            # Plot validation metric if available
            val_key = f'val_{metric_key}'
            if val_key in history_dict:
                ax.plot(epochs, history_dict[val_key], 'r--', label=f'Validation {metric_name}', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} over Epochs', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_summary(self, results_df, problem_type, save_path=None):
        """
        Create a comprehensive metrics summary visualization
        
        Args:
            results_df: DataFrame with model results
            problem_type: 'classification' or 'regression'
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Model Performance Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if problem_type == 'classification':
            metrics = ['accuracy', 'f1_score', 'roc_auc']
            metric_names = ['Accuracy', 'F1-Score', 'ROC-AUC']
        else:
            metrics = ['r2', 'rmse', 'mae']
            metric_names = ['R² Score', 'RMSE', 'MAE']
        
        # Top row: Individual metric bar charts
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            if metric in results_df.columns:
                ax = fig.add_subplot(gs[0, idx])
                sorted_df = results_df.sort_values(metric, ascending=(metric in ['rmse', 'mae']))
                
                bars = ax.bar(range(len(sorted_df)), sorted_df[metric], color='skyblue', edgecolor='navy')
                bars[0 if metric in ['rmse', 'mae'] else -1].set_color('lightgreen')
                
                ax.set_xticks(range(len(sorted_df)))
                ax.set_xticklabels(sorted_df['model_name'], rotation=45, ha='right', fontsize=9)
                ax.set_ylabel(name, fontsize=11)
                ax.set_title(f'{name} by Model', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
        
        # Middle row: Heatmap of all metrics
        ax_heatmap = fig.add_subplot(gs[1, :])
        metrics_data = results_df[['model_name'] + [m for m in metrics if m in results_df.columns]].set_index('model_name')
        
        # Normalize metrics for heatmap (0-1 scale)
        metrics_norm = metrics_data.copy()
        for col in metrics_norm.columns:
            if col in ['rmse', 'mae']:
                # For error metrics, lower is better
                metrics_norm[col] = 1 - (metrics_norm[col] - metrics_norm[col].min()) / (metrics_norm[col].max() - metrics_norm[col].min() + 1e-10)
            else:
                # For score metrics, higher is better
                metrics_norm[col] = (metrics_norm[col] - metrics_norm[col].min()) / (metrics_norm[col].max() - metrics_norm[col].min() + 1e-10)
        
        sns.heatmap(metrics_norm.T, annot=metrics_data.T, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Normalized Score'}, ax=ax_heatmap, linewidths=0.5)
        ax_heatmap.set_title('Metrics Heatmap (All Models)', fontsize=12, fontweight='bold')
        ax_heatmap.set_xlabel('')
        ax_heatmap.set_ylabel('Metrics', fontsize=11)
        
        # Bottom row: Best model highlight and statistics
        ax_best = fig.add_subplot(gs[2, :2])
        best_model = results_df.iloc[0]
        
        info_text = f"""
        🏆 BEST MODEL: {best_model['model_name'].upper()}
        
        Performance Metrics:
        """
        for metric, name in zip(metrics, metric_names):
            if metric in best_model:
                info_text += f"\n  • {name}: {best_model[metric]:.4f}"
        
        info_text += f"\n\n📊 Total Models Evaluated: {len(results_df)}"
        
        ax_best.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='center')
        ax_best.axis('off')
        
        # Bottom right: Model ranking
        ax_rank = fig.add_subplot(gs[2, 2])
        rankings = list(range(1, len(results_df) + 1))
        ax_rank.barh(results_df['model_name'], rankings, color='coral', edgecolor='darkred')
        ax_rank.set_xlabel('Rank (1 = Best)', fontsize=11)
        ax_rank.set_title('Model Rankings', fontsize=12, fontweight='bold')
        ax_rank.invert_xaxis()
        ax_rank.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics summary dashboard saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
