"""
Enhanced training metrics and progress tracking module.
"""

import time
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from .config import VIS_CONFIG, OUTPUTS_DIR

class TrainingTracker:
    def __init__(self, num_classes, class_names):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.num_classes = num_classes
        self.class_names = class_names
        self.metrics = {
            'training_time': 0,
            'epochs': [],
            'per_class_metrics': {},
            'confusion_matrices': {},
            'training_history': {},
            'system_info': self._get_system_info()
        }
        
    def _get_system_info(self):
        """Get system information for logging."""
        import platform
        import psutil
        import tensorflow as tf
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'tensorflow_version': tf.__version__,
            'gpu_available': bool(tf.config.list_physical_devices('GPU')),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def update_epoch_metrics(self, epoch, logs, model_name):
        """Update metrics after each epoch."""
        if model_name not in self.metrics['training_history']:
            self.metrics['training_history'][model_name] = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
        
        history = self.metrics['training_history'][model_name]
        history['loss'].append(logs.get('loss', 0))
        history['accuracy'].append(logs.get('accuracy', 0))
        history['val_loss'].append(logs.get('val_loss', 0))
        history['val_accuracy'].append(logs.get('val_accuracy', 0))
        
        self.metrics['epochs'].append(epoch)
    
    def update_class_metrics(self, y_true, y_pred, model_name):
        """Update per-class metrics."""
        report = classification_report(y_true, y_pred, 
                                    target_names=self.class_names, 
                                    output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        self.metrics['per_class_metrics'][model_name] = report
        self.metrics['confusion_matrices'][model_name] = conf_matrix.tolist()
    
    def plot_training_history(self, save_dir):
        """Plot training history for all models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, history in self.metrics['training_history'].items():
            # Plot accuracy
            plt.figure(figsize=VIS_CONFIG['plot_size'])
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} - Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f'{model_name}_accuracy.png', 
                       dpi=VIS_CONFIG['dpi'])
            plt.close()
            
            # Plot loss
            plt.figure(figsize=VIS_CONFIG['plot_size'])
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} - Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f'{model_name}_loss.png', 
                       dpi=VIS_CONFIG['dpi'])
            plt.close()
    
    def plot_confusion_matrices(self, save_dir):
        """Plot confusion matrices for all models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, conf_matrix in self.metrics['confusion_matrices'].items():
            plt.figure(figsize=VIS_CONFIG['plot_size'])
            sns.heatmap(conf_matrix, annot=True, fmt='d', 
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
            plt.title(f'{model_name} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(save_dir / f'{model_name}_confusion_matrix.png', 
                       dpi=VIS_CONFIG['dpi'])
            plt.close()
    
    def save_metrics(self, save_dir):
        """Save all metrics to JSON file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics['training_time'] = time.time() - self.start_time
        
        with open(save_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        self.logger.info(f"Saved training metrics to {save_dir}")
    
    def generate_report(self, save_dir):
        """Generate comprehensive training report."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# Plant Disease Classification - Training Report")
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System Information
        report.append("\n## System Information")
        for key, value in self.metrics['system_info'].items():
            report.append(f"- {key}: {value}")
        
        # Training Summary
        report.append("\n## Training Summary")
        report.append(f"- Total training time: {self.metrics['training_time']:.2f} seconds")
        report.append(f"- Number of epochs: {len(self.metrics['epochs'])}")
        
        # Model Performance
        report.append("\n## Model Performance")
        for model_name, metrics in self.metrics['per_class_metrics'].items():
            report.append(f"\n### {model_name}")
            report.append("\nPer-class metrics:")
            for class_name, class_metrics in metrics.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report.append(f"\n{class_name}:")
                    report.append(f"- Precision: {class_metrics['precision']:.4f}")
                    report.append(f"- Recall: {class_metrics['recall']:.4f}")
                    report.append(f"- F1-score: {class_metrics['f1-score']:.4f}")
        
        # Save report
        with open(save_dir / 'training_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info(f"Generated training report at {save_dir}")
