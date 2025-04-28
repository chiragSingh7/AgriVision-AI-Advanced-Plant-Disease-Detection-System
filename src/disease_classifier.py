"""
Disease classifier module for plant disease classification and severity analysis.
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from .config import VIS_CONFIG, OUTPUTS_DIR, LOGGING_CONFIG
from datetime import datetime

# Get logger instead of using basicConfig
logger = logging.getLogger(__name__)

class DiseaseClassifier:
    def __init__(self):
        """Initialize disease classifier with disease information."""
        self.logger = logging.getLogger(__name__)
        self.disease_info = {
            'bacterial_blight': {
                'name': 'Bacterial Blight',
                'severity_levels': {
                    'low': {'threshold': 0.1, 'color': (0, 255, 0)},
                    'medium': {'threshold': 0.3, 'color': (255, 255, 0)},
                    'high': {'threshold': 0.5, 'color': (255, 0, 0)}
                },
                'description': 'Bacterial blight is a common disease affecting rice plants.',
                'symptoms': ['Water-soaked lesions', 'Yellow halos', 'Leaf wilting'],
                'treatment': ['Use resistant varieties', 'Apply copper-based fungicides', 'Practice crop rotation']
            },
            'blast': {
                'name': 'Blast',
                'severity_levels': {
                    'low': {'threshold': 0.15, 'color': (0, 255, 0)},
                    'medium': {'threshold': 0.35, 'color': (255, 255, 0)},
                    'high': {'threshold': 0.6, 'color': (255, 0, 0)}
                },
                'description': 'Blast is a fungal disease that affects rice plants.',
                'symptoms': ['Diamond-shaped lesions', 'Gray centers', 'Reddish-brown margins'],
                'treatment': ['Use fungicides', 'Maintain proper spacing', 'Avoid excessive nitrogen']
            },
            'brown_spot': {
                'name': 'Brown Spot',
                'severity_levels': {
                    'low': {'threshold': 0.2, 'color': (0, 255, 0)},
                    'medium': {'threshold': 0.4, 'color': (255, 255, 0)},
                    'high': {'threshold': 0.7, 'color': (255, 0, 0)}
                },
                'description': 'Brown spot is a fungal disease affecting rice leaves.',
                'symptoms': ['Circular brown spots', 'Yellow halos', 'Leaf drying'],
                'treatment': ['Use resistant varieties', 'Apply fungicides', 'Maintain proper nutrition']
            }
        }
        
    def classify_disease(self, prediction_probs, threshold=0.5):
        """Classify disease based on prediction probabilities."""
        self.logger.info("Classifying disease")
        
        # Get the predicted class
        predicted_class = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_class]
        
        # Get disease information
        disease_key = list(self.disease_info.keys())[predicted_class]
        disease_data = self.disease_info[disease_key]
        
        return {
            'disease_name': disease_data['name'],
            'confidence': float(confidence),
            'description': disease_data['description'],
            'symptoms': disease_data['symptoms'],
            'treatment': disease_data['treatment']
        }
        
    def analyze_severity(self, img, mask):
        """Analyze disease severity based on affected area."""
        self.logger.info("Analyzing disease severity")
        
        # Calculate affected area ratio
        total_pixels = img.shape[0] * img.shape[1]
        affected_pixels = np.sum(mask > 0)
        affected_ratio = affected_pixels / total_pixels
        
        # Determine severity level
        severity_level = 'low'
        if affected_ratio > 0.5:
            severity_level = 'high'
        elif affected_ratio > 0.2:
            severity_level = 'medium'
            
        return {
            'severity_level': severity_level,
            'affected_ratio': float(affected_ratio),
            'affected_pixels': int(affected_pixels),
            'total_pixels': int(total_pixels)
        }
        
    def generate_report(self, img, prediction_probs, mask):
        """Generate comprehensive disease report."""
        self.logger.info("Generating disease report")
        
        # Classify disease
        disease_info = self.classify_disease(prediction_probs)
        
        # Analyze severity
        severity_info = self.analyze_severity(img, mask)
        
        # Combine information
        report = {
            'disease': disease_info,
            'severity': severity_info,
            'timestamp': datetime.now().isoformat(),
            'image_size': img.shape,
            'recommendations': self._generate_recommendations(
                disease_info['disease_name'],
                severity_info['severity_level']
            )
        }
        
        self.logger.info(f"Generated report for {disease_info['disease_name']}")
        return report
        
    def _generate_recommendations(self, disease_name, severity_level):
        """Generate recommendations based on disease and severity."""
        recommendations = []
        
        # Get disease-specific recommendations
        for disease_key, disease_data in self.disease_info.items():
            if disease_data['name'] == disease_name:
                recommendations.extend(disease_data['treatment'])
                break
                
        # Add severity-based recommendations
        if severity_level == 'high':
            recommendations.append('Immediate action required')
            recommendations.append('Consider consulting an expert')
        elif severity_level == 'medium':
            recommendations.append('Monitor closely')
            recommendations.append('Implement preventive measures')
        else:
            recommendations.append('Regular monitoring recommended')
            
        return recommendations
        
    def visualize_analysis(self, img, mask, report, save_path=None):
        """Visualize disease analysis results."""
        self.logger.info("Visualizing analysis results")
        
        # Create figure
        plt.figure(figsize=VIS_CONFIG["plot_size"])
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot mask with severity color
        severity_color = self.disease_info[list(self.disease_info.keys())[0]]['severity_levels'][report['severity']['severity_level']]['color']
        overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * np.array(severity_color) / 255, 0.3, 0)
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f"{report['disease']['disease_name']} - {report['severity']['severity_level'].title()} Severity")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_path is None:
            save_path = OUTPUTS_DIR / "disease_analysis.png"
        plt.savefig(save_path, dpi=VIS_CONFIG["dpi"])
        plt.close()
        
        return overlay 