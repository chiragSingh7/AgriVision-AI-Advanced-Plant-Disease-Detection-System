"""
Intersection over Union (IoU) calculator module for mask comparison.
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from .config import VIS_CONFIG, OUTPUTS_DIR, LOGGING_CONFIG

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class IoUCalculator:
    def __init__(self):
        """Initialize IoU calculator."""
        pass
        
    def calculate_iou(self, mask1, mask2):
        """Calculate Intersection over Union between two masks."""
        logger.info("Calculating IoU")
        
        # Ensure masks are binary
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        # Calculate IoU
        iou = np.sum(intersection) / np.sum(union)
        
        return float(iou)
        
    def calculate_precision_recall(self, mask1, mask2):
        """Calculate precision and recall between two masks."""
        logger.info("Calculating precision and recall")
        
        # Ensure masks are binary
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)
        
        # Calculate true positives, false positives, and false negatives
        true_positives = np.sum(np.logical_and(mask1, mask2))
        false_positives = np.sum(np.logical_and(mask1, np.logical_not(mask2)))
        false_negatives = np.sum(np.logical_and(np.logical_not(mask1), mask2))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        }
        
    def visualize_comparison(self, mask1, mask2, save_path=None):
        """Visualize comparison between two masks."""
        logger.info("Visualizing mask comparison")
        
        # Create visualization
        comparison = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
        
        # True positives (green)
        comparison[np.logical_and(mask1, mask2)] = [0, 255, 0]
        
        # False positives (red)
        comparison[np.logical_and(mask1, np.logical_not(mask2))] = [255, 0, 0]
        
        # False negatives (blue)
        comparison[np.logical_and(np.logical_not(mask1), mask2)] = [0, 0, 255]
        
        # Create figure
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        # Plot comparison
        plt.imshow(comparison)
        plt.title('Mask Comparison\n(Green: True Positives, Red: False Positives, Blue: False Negatives)')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = OUTPUTS_DIR / "mask_comparison.png"
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        
        return comparison
        
    def calculate_metrics(self, mask1, mask2):
        """Calculate comprehensive metrics between two masks."""
        logger.info("Calculating comprehensive metrics")
        
        # Calculate IoU
        iou = self.calculate_iou(mask1, mask2)
        
        # Calculate precision and recall
        metrics = self.calculate_precision_recall(mask1, mask2)
        
        # Add IoU to metrics
        metrics['iou'] = iou
        
        return metrics
        
    def batch_calculate_metrics(self, masks1, masks2):
        """Calculate metrics for a batch of mask pairs."""
        logger.info("Calculating metrics for batch of masks")
        
        metrics_list = []
        for mask1, mask2 in zip(masks1, masks2):
            metrics = self.calculate_metrics(mask1, mask2)
            metrics_list.append(metrics)
            
        # Calculate average metrics
        avg_metrics = {
            'iou': np.mean([m['iou'] for m in metrics_list]),
            'precision': np.mean([m['precision'] for m in metrics_list]),
            'recall': np.mean([m['recall'] for m in metrics_list]),
            'f1_score': np.mean([m['f1_score'] for m in metrics_list])
        }
        
        return {
            'individual_metrics': metrics_list,
            'average_metrics': avg_metrics
        } 