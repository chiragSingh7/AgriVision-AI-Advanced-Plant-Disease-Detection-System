"""
Main execution script for plant disease classification system.
"""

import os
import logging
import time
from pathlib import Path
from datetime import datetime
from src.config import OUTPUTS_DIR, LOGGING_CONFIG
from src.data_processing import DataProcessor
from src.feature_extractor import FeatureExtractor
from src.model_enhancements import EnhancedModelBuilder
from src.training_metrics import TrainingTracker
from src.visualization import Visualizer
from src.disease_analysis import DiseaseAnalyzer

def setup_logging():
    """Setup logging configuration."""
    # Create logs directory
    log_dir = OUTPUTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update log file path in config
    LOGGING_CONFIG["handlers"]["file"]["filename"] = str(
        log_dir / f"plant_disease_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(__name__)

def execute_pipeline():
    """Execute the complete classification pipeline."""
    # 1. Initial Setup
    logger = setup_logging()
    start_time = time.time()
    logger.info("Starting plant disease classification pipeline")
    
    try:
        # 2. Initialize Components
        logger.info("Initializing components...")
        data_processor = DataProcessor()
        feature_extractor = FeatureExtractor()
        model_builder = EnhancedModelBuilder()
        training_tracker = TrainingTracker()
        visualizer = Visualizer()
        disease_analyzer = DiseaseAnalyzer()
        
        # 3. Data Loading and Preprocessing
        logger.info("Loading and preprocessing data...")
        train_data, val_data, test_data = data_processor.load_and_preprocess_data()
        
        # 4. Feature Extraction
        logger.info("Extracting features...")
        train_features = feature_extractor.extract_all_features(train_data)
        val_features = feature_extractor.extract_all_features(val_data)
        test_features = feature_extractor.extract_all_features(test_data)
        
        # 5. Model Building and Training
        logger.info("Building and training model...")
        model = model_builder.build_enhanced_model()
        
        # Setup training callbacks
        callbacks = model_builder.get_callbacks()
        
        # Train model with progress tracking
        history = model.fit(
            train_features,
            validation_data=val_features,
            callbacks=callbacks,
            verbose=1
        )
        
        # 6. Model Evaluation
        logger.info("Evaluating model...")
        test_results = model.evaluate(test_features)
        training_tracker.update_metrics(history, test_results)
        
        # 7. Disease Analysis
        logger.info("Performing disease analysis...")
        disease_predictions = model.predict(test_features)
        disease_analysis = disease_analyzer.analyze_batch(
            test_data,
            disease_predictions
        )
        
        # 8. Visualization Generation
        logger.info("Generating visualizations...")
        visualizer.plot_training_history(history)
        visualizer.plot_confusion_matrix(test_data.labels, disease_predictions)
        visualizer.plot_feature_importance(feature_extractor.get_feature_importance())
        
        # 9. Results Saving
        logger.info("Saving results...")
        training_tracker.save_results()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
        
        return {
            'success': True,
            'execution_time': execution_time,
            'test_accuracy': test_results[1]
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Setup logging first
    logger = setup_logging()
    
    try:
        logger.info("Starting plant disease classification system")
        
        # Initialize components
        logger.info("Initializing components...")
        # ... rest of your code ...
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())