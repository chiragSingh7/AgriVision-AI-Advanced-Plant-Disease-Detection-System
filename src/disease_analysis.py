"""
Disease analysis module for severity assessment and IOU calculation.
"""

import numpy as np
import cv2
from skimage import measure
import logging
from pathlib import Path
from .config import LOGGING_CONFIG, OUTPUTS_DIR

# Setup module logger instead of basic config
logger = logging.getLogger(__name__)

class DiseaseAnalyzer:
    def __init__(self):
        """Initialize disease analyzer."""
        self.logger = logging.getLogger(__name__)
        # Disease severity levels
        self.severity_levels = {
            0: "Healthy",
            1: "Mild",
            2: "Moderate",
            3: "Severe"
        }
        
        # Disease information
        self.disease_info = {
            "bacterial_blight": {
                "name": "Bacterial Blight",
                "severity_thresholds": [0.1, 0.3, 0.5]  # Area thresholds for severity levels
            },
            "blast": {
                "name": "Blast",
                "severity_thresholds": [0.15, 0.35, 0.6]
            },
            "brown_spot": {
                "name": "Brown Spot",
                "severity_thresholds": [0.2, 0.4, 0.7]
            }
        }
        
    def calculate_iou(self, mask1, mask2):
        """Calculate Intersection over Union (IOU) between two masks."""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou
        
    def segment_disease(self, image):
        """Segment disease regions from the image."""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different diseases
        lower_brown = np.array([10, 100, 20])
        upper_brown = np.array([20, 255, 200])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
        
    def calculate_severity(self, image, disease_type):
        """Calculate disease severity level."""
        # Get disease thresholds
        thresholds = self.disease_info[disease_type]["severity_thresholds"]
        
        # Segment disease regions
        mask = self.segment_disease(image)
        
        # Calculate affected area ratio
        total_pixels = image.shape[0] * image.shape[1]
        affected_pixels = np.sum(mask > 0)
        affected_ratio = affected_pixels / total_pixels
        
        # Determine severity level
        if affected_ratio < thresholds[0]:
            severity = 0
        elif affected_ratio < thresholds[1]:
            severity = 1
        elif affected_ratio < thresholds[2]:
            severity = 2
        else:
            severity = 3
            
        return severity, affected_ratio
        
    def analyze_image(self, image, disease_type):
        """Analyze image for disease severity and features."""
        # Calculate severity
        severity_level, affected_ratio = self.calculate_severity(image, disease_type)
        
        # Get disease information
        disease_name = self.disease_info[disease_type]["name"]
        severity_name = self.severity_levels[severity_level]
        
        # Create analysis report
        report = {
            "disease_name": disease_name,
            "severity_level": severity_level,
            "severity_name": severity_name,
            "affected_ratio": affected_ratio,
            "recommendations": self._get_recommendations(severity_level)
        }
        
        return report
        
    def _get_recommendations(self, severity_level):
        """Get recommendations based on severity level."""
        recommendations = {
            0: "Plant is healthy. Continue regular monitoring.",
            1: "Mild infection detected. Monitor closely and consider preventive measures.",
            2: "Moderate infection. Apply appropriate treatment and increase monitoring frequency.",
            3: "Severe infection. Immediate treatment required. Consider removing affected parts."
        }
        return recommendations[severity_level]
        
    def visualize_analysis(self, image, mask, report, output_path):
        """Visualize disease analysis results."""
        # Create visualization
        vis_image = image.copy()
        
        # Overlay mask
        vis_image[mask > 0] = [255, 0, 0]  # Red overlay for affected areas
        
        # Add text
        text = f"{report['disease_name']} - {report['severity_name']}"
        cv2.putText(
            vis_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save visualization
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
    def compare_masks(self, mask1, mask2):
        """Compare two disease masks and calculate metrics."""
        # Calculate IOU
        iou = self.calculate_iou(mask1, mask2)
        
        # Calculate other metrics
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        precision = np.sum(intersection) / np.sum(mask2)
        recall = np.sum(intersection) / np.sum(mask1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        } 