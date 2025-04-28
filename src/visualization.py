"""
Enhanced visualization module for plant disease classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Union
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from src.config import VIS_CONFIG

class Visualizer:
    """Enhanced visualization class with detailed plotting capabilities."""
    
    def __init__(self):
        """Initialize visualizer with configuration settings."""
        self.logger = logging.getLogger(__name__)
        self.fig_size = VIS_CONFIG.get("figure_size", (12, 8))
        self.dpi = VIS_CONFIG.get("dpi", 100)
        self.cmap = VIS_CONFIG.get("colormap", "viridis")
        self.output_dir = Path(VIS_CONFIG.get("output_directory", "outputs/visualizations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, history: Dict, save_path: Union[str, Path] = None) -> None:
        """
        Plot training history with enhanced visualization.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        metrics = ['accuracy', 'loss']
        fig, axes = plt.subplots(1, 2, figsize=self.fig_size)
        
        for idx, metric in enumerate(metrics):
            axes[idx].plot(history[metric], label=f'Training {metric}')
            axes[idx].plot(history[f'val_{metric}'], label=f'Validation {metric}')
            axes[idx].set_title(f'Model {metric.capitalize()}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
            axes[idx].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved training history plot to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], save_path: Union[str, Path] = None) -> None:
        """
        Plot confusion matrix with enhanced visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=self.fig_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved confusion matrix plot to {save_path}")
        plt.close()
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str], 
                              save_path: Union[str, Path] = None) -> None:
        """
        Plot feature importance with enhanced visualization.
        
        Args:
            feature_importance: Array of feature importance scores
            feature_names: List of feature names
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        plt.barh(pos, feature_importance[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance Score')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved feature importance plot to {save_path}")
        plt.close()
    
    def plot_disease_severity(self, severity_scores: List[float], 
                            class_names: List[str], 
                            save_path: Union[str, Path] = None) -> None:
        """
        Plot disease severity scores with enhanced visualization.
        
        Args:
            severity_scores: List of disease severity scores
            class_names: List of class names
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        bars = plt.bar(range(len(severity_scores)), severity_scores)
        plt.xticks(range(len(severity_scores)), class_names, rotation=45, ha='right')
        plt.ylabel('Severity Score')
        plt.title('Disease Severity Analysis')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved disease severity plot to {save_path}")
        plt.close()
    
    def plot_sample_predictions(self, images: np.ndarray, predictions: np.ndarray, 
                              true_labels: np.ndarray, class_names: List[str],
                              save_path: Union[str, Path] = None) -> None:
        """
        Plot sample predictions with enhanced visualization.
        
        Args:
            images: Array of sample images
            predictions: Array of predicted probabilities
            true_labels: Array of true labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        n_samples = min(len(images), 5)
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        
        for i in range(n_samples):
            # Plot image
            axes[0, i].imshow(images[i])
            axes[0, i].axis('off')
            true_class = class_names[true_labels[i]]
            pred_class = class_names[np.argmax(predictions[i])]
            color = 'green' if true_class == pred_class else 'red'
            axes[0, i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                               color=color)
            
            # Plot prediction probabilities
            bars = axes[1, i].bar(range(len(class_names)), predictions[i])
            axes[1, i].set_xticks(range(len(class_names)))
            axes[1, i].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1, i].set_ylim([0, 1])
            
            # Highlight the predicted class
            bars[np.argmax(predictions[i])].set_color('red')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved sample predictions plot to {save_path}")
        plt.close()
    
    def plot_learning_curves(self, train_sizes: np.ndarray, 
                           train_scores: np.ndarray, 
                           val_scores: np.ndarray,
                           save_path: Union[str, Path] = None) -> None:
        """
        Plot learning curves with enhanced visualization.
        
        Args:
            train_sizes: Array of training sizes
            train_scores: Array of training scores
            val_scores: Array of validation scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, val_mean, label='Cross-validation score')
        
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved learning curves plot to {save_path}")
        plt.close()
    
    def plot_model_comparison(self, model_scores: Dict[str, float],
                            metric: str = 'accuracy',
                            save_path: Union[str, Path] = None) -> None:
        """
        Plot model comparison with enhanced visualization.
        
        Args:
            model_scores: Dictionary of model names and their scores
            metric: Name of the metric being compared
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        bars = plt.bar(models, scores)
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved model comparison plot to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, fpr: Dict[str, np.ndarray], 
                       tpr: Dict[str, np.ndarray],
                       roc_auc: Dict[str, float],
                       save_path: Union[str, Path] = None) -> None:
        """
        Plot ROC curves with enhanced visualization.
        
        Args:
            fpr: Dictionary of false positive rates for each class
            tpr: Dictionary of true positive rates for each class
            roc_auc: Dictionary of ROC AUC scores for each class
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.fig_size)
        
        for class_name in fpr.keys():
            plt.plot(fpr[class_name], tpr[class_name],
                    label=f'{class_name} (AUC = {roc_auc[class_name]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi)
            self.logger.info(f"Saved ROC curves plot to {save_path}")
        plt.close() 