"""
Enhanced training script for plant disease classification with detailed outputs and comprehensive heatmap analysis.
"""

import os
import logging.config
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
from src.config import LOGGING_CONFIG, MODEL_CONFIG, ML_CONFIG, VIS_CONFIG, OUTPUTS_DIR
from src.training_metrics import TrainingTracker
from src.model_enhancements import EnhancedModelBuilder

# Setup logging configuration first
def setup_logging():
    """Setup logging configuration."""
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)

# Setup logging before other imports
logger = setup_logging()

# Now import other modules
import numpy as np
import tensorflow as tf
from src.data_processing import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.visualization import Visualizer
from src.pca_analysis import PCAAnalyzer
from src.resnet_model import ResNetModel
from src.ml_classifiers import MLClassifiers
from src.heatmap import HeatmapGenerator
from src.disease_classifier import DiseaseClassifier
from src.iou_calculator import IoUCalculator

def print_section_header(logger, title):
    """Print formatted section header."""
    logger.info("\n" + "=" * 50)
    logger.info(f" {title} ")
    logger.info("=" * 50)

def save_layer_activations(heatmap_generator, img_batch, save_dir):
    """Save activation patterns for each convolutional layer."""
    activation_stats = heatmap_generator.analyze_activation_patterns(img_batch)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activation statistics
    with open(save_dir / "activation_stats.txt", "w") as f:
        for layer_name, stats in activation_stats.items():
            f.write(f"\nLayer: {layer_name}\n")
            f.write("-" * 30 + "\n")
            for stat_name, value in stats.items():
                if stat_name != 'shape':
                    f.write(f"{stat_name}: {value}\n")
                else:
                    f.write(f"{stat_name}: {list(value)}\n")

def generate_comparative_heatmaps(heatmap_generator, img_batch, predictions, class_names, save_dir):
    """Generate comparative heatmaps for correct and incorrect predictions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    true_labels = np.argmax(img_batch[1], axis=1)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Generate heatmaps for both correct and incorrect predictions
    for i, (img, true_label, pred_label) in enumerate(zip(img_batch[0], true_labels, pred_labels)):
        is_correct = true_label == pred_label
        status = "correct" if is_correct else "incorrect"
        
        # Generate heatmap
        heatmap = heatmap_generator.generate_grad_cam(
            img[np.newaxis, ...],
            class_idx=pred_label
        )
        
        # Save visualization with detailed information
        save_path = save_dir / f"heatmap_{status}_{i}.png"
        heatmap_generator.visualize_heatmap(
            img,
            heatmap,
            save_path=save_path,
            show_original=True,
            title=f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
        )

def load_and_prepare_data(data_processor):
    """Load and prepare data for training."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Loading and preparing data...")
        
        # Load data from raw directory
        data_dir = Path("data/raw")
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Create data generators with augmentation for training
        train_generator = data_processor.create_data_generator(
            str(data_dir),
            is_training=True
        )
        
        # Create validation generator
        val_generator = data_processor.create_data_generator(
            str(data_dir),
            is_training=False
        )
        
        # Create test generator
        test_generator = data_processor.create_data_generator(
            str(data_dir),
            is_training=False
        )
        
        logger.info(f"Found {train_generator.samples} training samples")
        logger.info(f"Found {val_generator.samples} validation samples")
        logger.info(f"Found {test_generator.samples} test samples")
        
        return train_generator, val_generator, test_generator
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def generate_comprehensive_report(output_dir, ml_results, resnet_metrics, class_names, start_time, heatmap_analysis=False):
    """Generate detailed training report with all metrics and results."""
    report_path = output_dir / "training_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("PLANT DISEASE CLASSIFICATION - TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Training information
        f.write("Training Information:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {(time.time() - start_time)/3600:.2f} hours\n\n")
        
        # Model configurations
        f.write("Model Configurations:\n")
        f.write("-" * 50 + "\n")
        f.write("ResNet Configuration:\n")
        for key, value in MODEL_CONFIG["resnet"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nML Classifiers Configuration:\n")
        for model, config in ML_CONFIG.items():
            f.write(f"\n{model}:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
        
        # Performance metrics
        f.write("\nPerformance Metrics:\n")
        f.write("-" * 50 + "\n")
        
        # ML Classifiers performance
        f.write("\n1. ML Classifiers Performance:\n")
        for model_name, report in ml_results.items():
            f.write(f"\n{model_name.replace('_', ' ').title()}:\n")
            f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")
            f.write("Per-class metrics:\n")
            for class_name in class_names:
                metrics = report[class_name]
                f.write(f"\n  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-score: {metrics['f1-score']:.4f}\n")
        
        # ResNet performance
        f.write("\n2. ResNet Model Performance:\n")
        f.write(f"Test Loss: {resnet_metrics[0]:.4f}\n")
        f.write(f"Test Accuracy: {resnet_metrics[1]:.4f}\n")
        
        # Generated files
        f.write("\nGenerated Files:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Model Files:\n")
        f.write("   - resnet_model.h5\n")
        f.write("   - random_forest_model.joblib\n")
        f.write("   - svm_model.joblib\n")
        f.write("   - logistic_regression_model.joblib\n")
        
        f.write("\n2. Visualization Files:\n")
        f.write("   - Training history plots\n")
        f.write("   - Confusion matrices\n")
        f.write("   - Grad-CAM heatmaps\n")
        f.write("   - Disease analysis visualizations\n")
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

def train_model(model, train_data, val_data, callbacks=None):
    """Train the model."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,  # You can adjust this
            callbacks=callbacks
        )
        logger.info("Model training completed")
        return history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def evaluate_model(model, test_data):
    """Evaluate the trained model."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Evaluating model...")
        results = model.evaluate(test_data)
        logger.info(f"Test accuracy: {results[1]:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    """Main training function."""
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("Starting training process...")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU(s) available: {len(gpus)}")
        
        # Initialize components
        data_processor = DataProcessor()
        feature_extractor = FeatureExtractor()
        ml_classifiers = MLClassifiers()
        resnet_model = ResNetModel(num_classes=4)  # Adjust based on your classes
        
        # Load and prepare data
        train_data, val_data, test_data = load_and_prepare_data(data_processor)
        
        # Build and compile model
        model = resnet_model.build()
        resnet_model.compile()
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(OUTPUTS_DIR / "models" / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(OUTPUTS_DIR / "logs" / "tensorboard"),
                histogram_freq=1
            )
        ]
        
        # Train model
        history = train_model(model, train_data, val_data, callbacks)
        
        # Evaluate model
        results = evaluate_model(model, test_data)
        
        logger.info("Training pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
