"""
Enhanced evaluation script for plant disease classification with comprehensive heatmap analysis.
"""

import os
import numpy as np
import tensorflow as tf
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.data_processing import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.visualization import Visualizer
from src.ml_classifiers import MLClassifiers
from src.heatmap import HeatmapGenerator
from src.disease_classifier import DiseaseClassifier
from src.iou_calculator import IoUCalculator
from src.config import LOGGING_CONFIG, MODEL_CONFIG, ML_CONFIG, VIS_CONFIG, OUTPUTS_DIR

def setup_logging():
    """Setup enhanced logging with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    LOGGING_CONFIG["handlers"]["file"]["filename"] = str(log_dir / f"evaluation_{timestamp}.log")
    logging.basicConfig(**LOGGING_CONFIG)
    return logging.getLogger(__name__)

def print_section_header(logger, title):
    """Print formatted section header."""
    logger.info("\n" + "=" * 50)
    logger.info(f" {title} ")
    logger.info("=" * 50)

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix with enhanced visualization."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model_performance(y_true, y_pred, class_names):
    """Evaluate model performance with detailed metrics."""
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names, 
                                 output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'accuracy': report['accuracy'],
        'macro_avg': report['macro avg'],
        'weighted_avg': report['weighted avg']
    }
    
    return metrics

def analyze_heatmap_correlation(heatmaps, disease_masks, predictions, severity_scores):
    """Analyze correlation between heatmaps, disease regions, and severity scores."""
    correlations = {
        'heatmap_mask_iou': [],
        'heatmap_severity': [],
        'confidence_severity': []
    }
    
    for heatmap, mask, pred, severity in zip(heatmaps, disease_masks, predictions, severity_scores):
        # Calculate IoU between heatmap and disease mask
        heatmap_binary = (heatmap > 0.5).astype(np.uint8)
        iou = np.sum(heatmap_binary & mask) / np.sum(heatmap_binary | mask)
        correlations['heatmap_mask_iou'].append(iou)
        
        # Calculate correlation between heatmap intensity and severity
        heatmap_mean = np.mean(heatmap)
        correlations['heatmap_severity'].append((heatmap_mean, severity))
        
        # Calculate correlation between prediction confidence and severity
        confidence = np.max(pred)
        correlations['confidence_severity'].append((confidence, severity))
    
    return correlations

def analyze_misclassifications(images, true_labels, predictions, heatmaps, class_names, save_dir):
    """Analyze misclassified samples with detailed heatmap visualization."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    misclassified_analysis = {
        'total_samples': len(predictions),
        'misclassified_count': 0,
        'class_confusion': {},
        'confidence_distribution': [],
        'heatmap_characteristics': []
    }
    
    pred_classes = np.argmax(predictions, axis=1)
    
    for i, (img, true_label, pred, heatmap) in enumerate(zip(images, true_labels, predictions, heatmaps)):
        pred_class = pred_classes[i]
        confidence = np.max(pred)
        
        if true_label != pred_class:
            misclassified_analysis['misclassified_count'] += 1
            
            # Record confusion pair
            confusion_pair = f"{class_names[true_label]}->{class_names[pred_class]}"
            misclassified_analysis['class_confusion'][confusion_pair] = \
                misclassified_analysis['class_confusion'].get(confusion_pair, 0) + 1
            
            # Analyze heatmap characteristics
            heatmap_stats = {
                'mean_intensity': np.mean(heatmap),
                'max_intensity': np.max(heatmap),
                'coverage': np.sum(heatmap > 0.5) / heatmap.size
            }
            misclassified_analysis['heatmap_characteristics'].append(heatmap_stats)
            
            # Save detailed visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap overlay
            axes[1].imshow(img)
            axes[1].imshow(heatmap, alpha=0.6, cmap='jet')
            axes[1].set_title('Heatmap Overlay')
            axes[1].axis('off')
            
            # Prediction probabilities
            bars = axes[2].bar(range(len(class_names)), pred)
            axes[2].set_xticks(range(len(class_names)))
            axes[2].set_xticklabels(class_names, rotation=45, ha='right')
            axes[2].set_title('Class Probabilities')
            
            # Highlight true and predicted classes
            bars[true_label].set_color('green')
            bars[pred_class].set_color('red')
            
            plt.tight_layout()
            plt.savefig(save_dir / f"misclassified_{i}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            misclassified_analysis['confidence_distribution'].append(confidence)
    
    return misclassified_analysis

def main():
    """Enhanced main evaluation function with comprehensive heatmap analysis."""
    logger = setup_logging()
    start_time = time.time()
    
    try:
        print_section_header(logger, "PLANT DISEASE CLASSIFICATION EVALUATION")
        
        # System information
        logger.info("\nSystem Information:")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
        logger.info(f"Evaluation start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize components
        print_section_header(logger, "INITIALIZING COMPONENTS")
        components = {
            "Data Processor": DataProcessor(),
            "Feature Extractor": FeatureExtractor(),
            "Visualizer": Visualizer(),
            "Disease Classifier": DiseaseClassifier(),
            "IoU Calculator": IoUCalculator(),
            "ML Classifiers": MLClassifiers(random_state=ML_CONFIG["random_forest"]["random_state"])
        }
        
        for name, component in components.items():
            logger.info(f"✓ Initialized {name}")
        
        # Load test data
        print_section_header(logger, "LOADING TEST DATA")
        with tqdm(total=1, desc="Loading test data") as pbar:
            test_data = components["Data Processor"].load_test_data("data/test")
            pbar.update(1)
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Feature extraction
        print_section_header(logger, "FEATURE EXTRACTION")
        with tqdm(total=1, desc="Extracting features") as pbar:
            X_test, y_test = components["Feature Extractor"].extract_all_features(test_data)
            pbar.update(1)
        
        # Load models
        print_section_header(logger, "LOADING MODELS")
        models_dir = OUTPUTS_DIR / "models"
        
        # Load ML Classifiers
        ml_classifiers = components["ML Classifiers"]
        ml_classifiers.load_models(models_dir)
        
        # Load ResNet model
        resnet_model = tf.keras.models.load_model(str(models_dir / "resnet_model.h5"))
        logger.info("✓ Loaded all models successfully")
        
        # Initialize HeatmapGenerator
        heatmap_generator = HeatmapGenerator(resnet_model)
        
        # Evaluate ML Classifiers
        print_section_header(logger, "EVALUATING ML CLASSIFIERS")
        ml_results = {}
        
        for model_name in ["random_forest", "svm", "logistic_regression", "voting"]:
            logger.info(f"\nEvaluating {model_name}...")
            y_pred = ml_classifiers.predict(model_name, X_test)
            metrics = evaluate_model_performance(y_test, y_pred, 
                                              components["Data Processor"].class_names)
            ml_results[model_name] = metrics
            
            # Save confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                components["Data Processor"].class_names,
                OUTPUTS_DIR / "visualizations" / f"confusion_matrix_{model_name}.png"
            )
        
        # Evaluate ResNet
        print_section_header(logger, "EVALUATING RESNET MODEL")
        test_generator = components["Data Processor"].create_data_generator(test_data, 
                                                                         is_training=False)
        resnet_metrics = resnet_model.evaluate(test_generator, verbose=1)
        y_pred_resnet = resnet_model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred_resnet, axis=1)
        
        resnet_results = evaluate_model_performance(
            test_generator.classes,
            y_pred_classes,
            components["Data Processor"].class_names
        )
        
        # Comprehensive heatmap analysis
        print_section_header(logger, "COMPREHENSIVE HEATMAP ANALYSIS")
        
        # Generate heatmaps for all test images
        all_images = []
        all_heatmaps = []
        all_masks = []
        all_predictions = []
        severity_scores = []
        
        with tqdm(total=len(test_data), desc="Generating heatmaps and analysis") as pbar:
            for i, (img, label) in enumerate(test_data):
                # Generate heatmap
                heatmap = heatmap_generator.generate_grad_cam(
                    img[np.newaxis, ...],
                    class_idx=y_pred_classes[i]
                )
                
                # Get disease mask and severity score
                mask = components["Feature Extractor"].segment_disease(img)
                severity = components["Disease Classifier"].calculate_severity(mask)
                
                all_images.append(img)
                all_heatmaps.append(heatmap)
                all_masks.append(mask)
                all_predictions.append(y_pred_resnet[i])
                severity_scores.append(severity)
                
                pbar.update(1)
        
        # Analyze correlations
        correlations = analyze_heatmap_correlation(
            all_heatmaps,
            all_masks,
            all_predictions,
            severity_scores
        )
        
        # Analyze misclassifications
        misclassified_analysis = analyze_misclassifications(
            all_images,
            test_generator.classes,
            all_predictions,
            all_heatmaps,
            components["Data Processor"].class_names,
            OUTPUTS_DIR / "visualizations" / "misclassified"
        )
        
        # Generate comprehensive evaluation report
        print_section_header(logger, "GENERATING EVALUATION REPORT")
        generate_evaluation_report(
            OUTPUTS_DIR / "reports",
            ml_results,
            resnet_results,
            resnet_metrics,
            correlations,
            misclassified_analysis,
            components["Data Processor"].class_names,
            start_time
        )
        
        # Evaluation complete
        end_time = time.time()
        evaluation_time = end_time - start_time
        print_section_header(logger, "EVALUATION COMPLETE")
        logger.info(f"Total evaluation time: {evaluation_time:.2f} seconds")
        logger.info(f"Results saved in: {OUTPUTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

def generate_evaluation_report(output_dir, ml_results, resnet_results, resnet_metrics,
                             correlations, misclassified_analysis, class_names, start_time):
    """Generate comprehensive evaluation report with heatmap analysis."""
    report_path = output_dir / "evaluation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("PLANT DISEASE CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Evaluation information
        f.write("Evaluation Information:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {(time.time() - start_time):.2f} seconds\n\n")
        
        # Model Performance
        f.write("Model Performance:\n")
        f.write("-" * 50 + "\n")
        
        # ML Classifiers
        f.write("\nML Classifiers Performance:\n")
        for model_name, results in ml_results.items():
            f.write(f"\n{model_name.replace('_', ' ').title()}:\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write("\nPer-class metrics:\n")
            for class_name in class_names:
                if class_name in results['classification_report']:
                    metrics = results['classification_report'][class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
        
        # ResNet Performance
        f.write("\nResNet Model Performance:\n")
        f.write(f"Test Loss: {resnet_metrics[0]:.4f}\n")
        f.write(f"Test Accuracy: {resnet_metrics[1]:.4f}\n")
        
        # Heatmap Analysis
        f.write("\nHeatmap Analysis:\n")
        f.write("-" * 50 + "\n")
        
        # Correlation Analysis
        f.write("\nCorrelation Analysis:\n")
        for metric, values in correlations.items():
            if metric == 'heatmap_mask_iou':
                mean_iou = np.mean(values)
                f.write(f"Mean IoU between heatmaps and disease masks: {mean_iou:.4f}\n")
            else:
                corr, p_value = pearsonr(*zip(*values))
                f.write(f"{metric} correlation: {corr:.4f} (p-value: {p_value:.4f})\n")
        
        # Misclassification Analysis
        f.write("\nMisclassification Analysis:\n")
        f.write(f"Total samples: {misclassified_analysis['total_samples']}\n")
        f.write(f"Misclassified samples: {misclassified_analysis['misclassified_count']}\n")
        f.write(f"Error rate: {misclassified_analysis['misclassified_count']/misclassified_analysis['total_samples']:.4f}\n")
        
        f.write("\nClass Confusion Pairs:\n")
        for pair, count in sorted(misclassified_analysis['class_confusion'].items(), 
                                key=lambda x: x[1], reverse=True):
            f.write(f"{pair}: {count}\n")
        
        if misclassified_analysis['confidence_distribution']:
            mean_conf = np.mean(misclassified_analysis['confidence_distribution'])
            f.write(f"\nMean confidence for misclassified samples: {mean_conf:.4f}\n")
        
        if misclassified_analysis['heatmap_characteristics']:
            mean_intensity = np.mean([h['mean_intensity'] for h in misclassified_analysis['heatmap_characteristics']])
            mean_coverage = np.mean([h['coverage'] for h in misclassified_analysis['heatmap_characteristics']])
            f.write(f"Mean heatmap intensity for misclassified samples: {mean_intensity:.4f}\n")
            f.write(f"Mean heatmap coverage for misclassified samples: {mean_coverage:.4f}\n")
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU memory growth error: {str(e)}")
    
    main()
