"""
Enhanced model builder module for plant disease classification.
Handles model creation, training, and evaluation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
import numpy as np
import logging
from pathlib import Path
from src.config import (
    MODEL_CONFIG, OUTPUTS_DIR, LOGGING_CONFIG
)
from typing import Tuple, Optional, Dict, Any

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ModelBuilder:
    """Enhanced model builder class with detailed configuration and logging."""
    
    def __init__(self):
        """Initialize model builder with configuration settings."""
        self.logger = logging.getLogger(__name__)
        self.input_shape = tuple(MODEL_CONFIG["input_shape"])
        self.num_classes = MODEL_CONFIG["num_classes"]
        self.learning_rate = MODEL_CONFIG["learning_rate"]
        self.dropout_rate = MODEL_CONFIG["dropout_rate"]
        self.l2_lambda = MODEL_CONFIG["l2_lambda"]
        
        # ResNet specific configurations
        self.resnet_config = MODEL_CONFIG["resnet"]
        self.fine_tune_layers = self.resnet_config.get("fine_tune_layers", 0)
        self.pooling = self.resnet_config.get("pooling", "avg")
        
        self.logger.info("Initialized ModelBuilder with configurations:")
        self.logger.info(f"Input shape: {self.input_shape}")
        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
    
    def build_resnet_model(self, weights: str = 'imagenet') -> tf.keras.Model:
        """
        Build and return an enhanced ResNet model.
        
        Args:
            weights: Pre-trained weights to use
            
        Returns:
            Compiled ResNet model
        """
        self.logger.info("Building ResNet model...")
        
        # Base model
        base_model = applications.ResNet50V2(
            weights=weights,
            include_top=False,
            input_shape=self.input_shape,
            pooling=self.pooling
        )
        
        # Freeze base model layers
        base_model.trainable = False
        if self.fine_tune_layers > 0:
            for layer in base_model.layers[-self.fine_tune_layers:]:
                layer.trainable = True
        
        # Build model
        model = models.Sequential([
            base_model,
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(512, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("ResNet model built successfully")
        self.logger.info(f"Total layers: {len(model.layers)}")
        self.logger.info(f"Trainable weights: {len(model.trainable_weights)}")
        
        return model
    
    def build_custom_cnn(self) -> tf.keras.Model:
        """
        Build and return an enhanced custom CNN model.
        
        Returns:
            Compiled custom CNN model
        """
        self.logger.info("Building custom CNN model...")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_lambda)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("Custom CNN model built successfully")
        self.logger.info(f"Total layers: {len(model.layers)}")
        self.logger.info(f"Trainable weights: {len(model.trainable_weights)}")
        
        return model
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a saved model with proper error handling.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            self.logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model: tf.keras.Model, save_path: str) -> None:
        """
        Save a model with proper error handling.
        
        Args:
            model: Model to save
            save_path: Path to save the model
        """
        try:
            self.logger.info(f"Saving model to {save_path}")
            model.save(save_path)
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model_summary(self, model: tf.keras.Model) -> str:
        """
        Get a detailed string summary of the model.
        
        Args:
            model: Model to summarize
            
        Returns:
            String containing model summary
        """
        # Create string buffer to capture summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        
        # Add additional information
        summary_list.append("\nModel Configuration:")
        summary_list.append(f"Input Shape: {self.input_shape}")
        summary_list.append(f"Number of Classes: {self.num_classes}")
        summary_list.append(f"Learning Rate: {self.learning_rate}")
        summary_list.append(f"Dropout Rate: {self.dropout_rate}")
        summary_list.append(f"L2 Regularization: {self.l2_lambda}")
        
        if hasattr(model, 'optimizer'):
            summary_list.append(f"\nOptimizer: {model.optimizer.__class__.__name__}")
            summary_list.append(f"Loss Function: {model.loss}")
            summary_list.append(f"Metrics: {model.metrics_names}")
        
        return "\n".join(summary_list)
    
    def get_layer_info(self, model: tf.keras.Model) -> Dict[str, Any]:
        """
        Get detailed information about model layers.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary containing layer information
        """
        layer_info = {
            'total_layers': len(model.layers),
            'trainable_params': 0,
            'non_trainable_params': 0,
            'layer_types': {},
            'layers': []
        }
        
        for layer in model.layers:
            # Count parameters
            trainable_count = int(
                sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
            )
            non_trainable_count = int(
                sum([tf.keras.backend.count_params(w) for w in layer.non_trainable_weights])
            )
            
            layer_info['trainable_params'] += trainable_count
            layer_info['non_trainable_params'] += non_trainable_count
            
            # Count layer types
            layer_type = layer.__class__.__name__
            if layer_type not in layer_info['layer_types']:
                layer_info['layer_types'][layer_type] = 0
            layer_info['layer_types'][layer_type] += 1
            
            # Store layer details
            layer_info['layers'].append({
                'name': layer.name,
                'type': layer_type,
                'trainable': layer.trainable,
                'trainable_params': trainable_count,
                'non_trainable_params': non_trainable_count,
                'output_shape': str(layer.output_shape)
            })
        
        return layer_info
        
    def compile_model(self):
        """Compile the model with appropriate optimizer and loss."""
        logger.info("Compiling model")
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=MODEL_CONFIG["learning_rate"]
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train_model(self, train_data, val_data, callbacks=None):
        """Train the model with callbacks."""
        logger.info("Training model")
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
            
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=MODEL_CONFIG["epochs"],
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def _get_default_callbacks(self):
        """Get default training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=MODEL_CONFIG["early_stopping_patience"],
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=str(OUTPUTS_DIR / "best_model.h5"),
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            TensorBoard(
                log_dir=str(OUTPUTS_DIR / "logs"),
                histogram_freq=1
            )
        ]
        
        return callbacks
        
    def evaluate_model(self, test_data):
        """Evaluate the model on test data."""
        logger.info("Evaluating model")
        
        results = self.model.evaluate(test_data, verbose=1)
        return dict(zip(self.model.metrics_names, results))
        
    def predict(self, data):
        """Make predictions using the model."""
        logger.info("Making predictions")
        
        predictions = self.model.predict(data, verbose=1)
        return predictions 