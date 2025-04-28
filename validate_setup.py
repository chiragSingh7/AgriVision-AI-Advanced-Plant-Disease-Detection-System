"""
Validation script to ensure all components are properly setup and connected.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
import cv2
import sklearn
import matplotlib.pyplot as plt

def validate_environment():
    """Validate Python environment and dependencies."""
    try:
        # Check Python version
        python_version = sys.version_info
        assert python_version.major == 3 and python_version.minor >= 8
        
        # Check TensorFlow
        assert tf.__version__ >= "2.8.0"
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Found {len(gpus)} GPU devices")
        
        # Check TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        return True
    except AssertionError as e:
        print(f"Environment validation failed: {str(e)}")
        return False
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        return False
    except Exception as e:
        print(f"Environment error: {str(e)}")
        return False

def validate_data():
    """Validate data directory structure and files."""
    try:
        required_dirs = [
            'data/raw/bacterial_blight',
            'data/raw/blast',
            'data/raw/brown_spot',
            'data/raw/healthy',
            'data/processed',
            'outputs/logs',
            'outputs/models',
            'outputs/visualizations'
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print("Missing directories:")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
            return False
        
        # Check for image files
        for class_dir in Path('data/raw').iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg'))
                if not images:
                    print(f"No images found in {class_dir}")
                    return False
        
        return True
    except Exception as e:
        print(f"Data validation error: {str(e)}")
        return False

def validate_models():
    """Validate model configurations and architectures."""
    try:
        from src.model_enhancements import EnhancedModelBuilder
        from src.config import MODEL_CONFIG
        
        # Test model building
        model_builder = EnhancedModelBuilder(num_classes=4)
        model = model_builder.build_enhanced_model()
        
        # Validate model architecture
        assert len(model.layers) > 0
        assert model.output_shape[-1] == 4
        
        return True
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False

def validate_pipeline():
    """Validate complete pipeline with small dataset."""
    try:
        from src.data_processing import DataProcessor
        from src.feature_extractor import FeatureExtractor
        
        # Test with small dataset
        data_processor = DataProcessor()
        feature_extractor = FeatureExtractor()
        
        # Load small batch of data
        sample_data = data_processor.load_sample_data(n_samples=5)
        
        # Extract features
        features = feature_extractor.extract_all_features(sample_data)
        
        assert features.shape[0] == 5
        
        return True
    except Exception as e:
        print(f"Pipeline validation failed: {str(e)}")
        return False

def main():
    """Run all validation checks."""
    validations = {
        "Environment": validate_environment,
        "Data": validate_data,
        "Models": validate_models,
        "Pipeline": validate_pipeline
    }
    
    results = {}
    all_passed = True
    
    print("Starting validation checks...")
    for name, validation_func in validations.items():
        print(f"\nValidating {name}...")
        try:
            passed = validation_func()
            results[name] = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
        except Exception as e:
            results[name] = f"ERROR: {str(e)}"
            all_passed = False
    
    # Print results
    print("\nValidation Results:")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name}: {result}")
    print("=" * 50)
    print(f"\nOverall Status: {'PASSED' if all_passed else 'FAILED'}")

if __name__ == "__main__":
    main()
