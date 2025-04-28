"""
Configuration settings for the plant disease classification project.
"""

from pathlib import Path
import os

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    "class_names": [
        "bacterial_blight",
        "blast",
        "brown_spot",
        "healthy"
    ],
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": 0.2,
        "shear_range": 0.2,
        "fill_mode": "nearest",
        "brightness_range": [0.8, 1.2]
    }
}

# Model configuration
MODEL_CONFIG = {
    "epochs": 100,
    "learning_rate": 0.0001,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 8,
    "reduce_lr_factor": 0.2,
    "batch_size": 32,
    "use_mixed_precision": True,
    "use_gradient_clipping": True,
    "label_smoothing": 0.1,
    "warmup_epochs": 5,
    "use_cross_validation": True,
    "n_folds": 5,
    "resnet": {
        "weights": "imagenet",
        "include_top": False,
        "input_shape": (224, 224, 3),
        "dense_units": [2048, 1024],
        "dropout_rate": 0.3,
        "l2_lambda": 0.0001,
        "attention_units": 512
    }
}

# ML Classifiers configuration
ML_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    },
    "svm": {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "auto",
        "probability": True,
        "class_weight": "balanced",
        "random_state": 42
    },
    "logistic_regression": {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42
    },
    "voting": {
        "voting": "soft",
        "weights": [3, 2, 1]
    }
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "color_features": True,
    "texture_features": True,
    "shape_features": True,
    "hog_features": True,
    "lbp_features": True,
    "deep_features": True,
    "use_multiple_scales": True,
    "scales": [1.0, 0.75, 0.5],
    "use_pca": True,
    "pca_components": 100,
    "feature_selection": True,
    "n_selected_features": 500
}

# Visualization configuration
VIS_CONFIG = {
    "plot_size": (12, 8),
    "dpi": 300,
    "cmap": "viridis",
    "font_size": 12,
    "title_font_size": 14,
    "label_font_size": 12,
    "tick_font_size": 10
}

# Disease severity configuration
SEVERITY_CONFIG = {
    "low_threshold": 0.2,
    "medium_threshold": 0.4,
    "high_threshold": 0.6,
    "severity_colors": {
        "low": (0, 255, 0),    # Green
        "medium": (255, 255, 0),  # Yellow
        "high": (255, 0, 0)    # Red
    }
}

# IoU configuration
IOU_CONFIG = {
    "threshold": 0.5,
    "visualization_colors": {
        "true_positive": (0, 255, 0),    # Green
        "false_positive": (255, 0, 0),   # Red
        "false_negative": (0, 0, 255)    # Blue
    }
}

# Data Augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2],
    "noise_stddev": 0.01,
    "translation_range": 0.1
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOGS_DIR / "plant_disease.log"),
            "mode": "a"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
} 