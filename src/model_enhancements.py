"""
Enhanced model architectures and training strategies for improved accuracy.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Activation, Multiply, RandomRotation, RandomZoom, RandomFlip,
    RandomTranslation, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import logging
from .config import MODEL_CONFIG, AUGMENTATION_CONFIG, OUTPUTS_DIR

class EnhancedModelBuilder:
    def __init__(self, num_classes):
        self.logger = logging.getLogger(__name__)
        self.num_classes = num_classes
        self.config = MODEL_CONFIG
        
    def build_enhanced_model(self):
        """Build enhanced model with attention and regularization."""
        try:
            # Base ResNet model
            base_model = ResNet50V2(
                weights=self.config["resnet"]["weights"],
                include_top=False,
                input_shape=self.config["resnet"]["input_shape"]
            )
            
            # Freeze early layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            
            # Add custom layers
            x = base_model.output
            x = GlobalAveragePooling2D(name='gap')(x)
            
            # Multiple dense layers with batch normalization
            for i, units in enumerate(self.config["resnet"]["dense_units"]):
                x = Dense(units, 
                         kernel_regularizer=l2(self.config["resnet"]["l2_lambda"]),
                         name=f'dense_{i}')(x)
                x = BatchNormalization(name=f'bn_{i}')(x)
                x = Activation('relu', name=f'relu_{i}')(x)
                x = Dropout(self.config["resnet"]["dropout_rate"], 
                          name=f'dropout_{i}')(x)
            
            # Attention mechanism
            attention = Dense(self.config["resnet"]["attention_units"], 
                            activation='tanh', 
                            name='attention_1')(x)
            attention = Dense(1, activation='sigmoid', name='attention_2')(attention)
            x = Multiply(name='attention_multiply')([x, attention])
            
            # Output layer
            outputs = Dense(self.num_classes, 
                          activation='softmax', 
                          name='output')(x)
            
            model = Model(base_model.input, outputs, name='enhanced_resnet')
            self.logger.info("Successfully built enhanced model")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
    
    def get_callbacks(self, fold=None):
        """Get training callbacks with enhanced monitoring."""
        try:
            suffix = f"_fold_{fold}" if fold is not None else ""
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config["early_stopping_patience"],
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config["reduce_lr_factor"],
                    patience=self.config["reduce_lr_patience"],
                    min_lr=1e-6,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(OUTPUTS_DIR / "models" / f'best_model{suffix}.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(OUTPUTS_DIR / "logs" / f'training{suffix}'),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=0
                )
            ]
            return callbacks
            
        except Exception as e:
            self.logger.error(f"Error creating callbacks: {str(e)}")
            raise
    
    def get_augmentation_layer(self):
        """Create data augmentation layer."""
        try:
            return tf.keras.Sequential([
                RandomRotation(
                    AUGMENTATION_CONFIG["rotation_range"] / 360,
                    name='random_rotation'
                ),
                RandomZoom(
                    AUGMENTATION_CONFIG["zoom_range"],
                    name='random_zoom'
                ),
                RandomFlip(
                    "horizontal_and_vertical",
                    name='random_flip'
                ),
                RandomTranslation(
                    AUGMENTATION_CONFIG["translation_range"],
                    AUGMENTATION_CONFIG["translation_range"],
                    name='random_translation'
                ),
                GaussianNoise(
                    AUGMENTATION_CONFIG["noise_stddev"],
                    name='gaussian_noise'
                )
            ], name='augmentation_layer')
            
        except Exception as e:
            self.logger.error(f"Error creating augmentation layer: {str(e)}")
            raise
    
    def compile_model(self, model):
        """Compile model with optimized settings."""
        try:
            if self.config["use_mixed_precision"]:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config["learning_rate"]
            )
            
            if self.config["use_gradient_clipping"]:
                optimizer.clipnorm = 1.0
            
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=self.config["label_smoothing"]
                ),
                metrics=['accuracy']
            )
            return model
            
        except Exception as e:
            self.logger.error(f"Error compiling model: {str(e)}")
            raise
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """Train model with cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            # Build and compile model
            model = self.build_enhanced_model()
            model = self.compile_model(model)
            
            # Train model
            history = model.fit(
                X[train_idx], y[train_idx],
                validation_data=(X[val_idx], y[val_idx]),
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                callbacks=self.get_callbacks(fold)
            )
            
            # Evaluate
            score = model.evaluate(X[val_idx], y[val_idx])
            scores.append(score[1])  # Accuracy
            print(f"Fold {fold + 1} Accuracy: {score[1]:.4f}")
        
        print(f"\nMean CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return scores
