"""
ResNet model implementation for plant disease classification.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import logging
from pathlib import Path
from .config import MODEL_CONFIG, OUTPUTS_DIR

# Get logger instead of using basicConfig
logger = logging.getLogger(__name__)

class ResNetModel:
    def __init__(self, num_classes):
        """Initialize ResNet model."""
        self.num_classes = num_classes
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def build(self):
        """Build ResNet model with custom top layers."""
        try:
            # Base ResNet model
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=MODEL_CONFIG["resnet"]["input_shape"]
            )
            
            # Add custom top layers
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(MODEL_CONFIG["resnet"]["dense_units"], activation='relu')(x)
            x = Dropout(MODEL_CONFIG["resnet"]["dropout_rate"])(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            self.model = Model(inputs=base_model.input, outputs=predictions)
            self.logger.info("ResNet model built successfully")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error building ResNet model: {str(e)}")
            raise
    
    def compile(self):
        """Compile the model."""
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(MODEL_CONFIG["learning_rate"]),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.logger.info("Model compiled successfully")
            
        except Exception as e:
            self.logger.error(f"Error compiling model: {str(e)}")
            raise
    
    def train(self, train_data, val_data, epochs=None, callbacks=None):
        """Train the model."""
        try:
            if epochs is None:
                epochs = MODEL_CONFIG["epochs"]
                
            history = self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks
            )
            
            self.logger.info("Model training completed")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def save(self, filepath=None):
        """Save the model."""
        try:
            if filepath is None:
                filepath = OUTPUTS_DIR / "models" / "resnet_model.h5"
                
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
        
    def evaluate(self, test_generator):
        """Evaluate the model."""
        logger.info("Evaluating ResNet model")
        return self.model.evaluate(test_generator)
        
    def predict(self, X):
        """Make predictions."""
        logger.info("Making predictions with ResNet")
        return self.model.predict(X)
        
    def load_model(self, path):
        """Load a saved model."""
        logger.info(f"Loading model from {path}")
        self.model = tf.keras.models.load_model(path)
        
    def unfreeze_layers(self, num_layers=20):
        """Unfreeze some layers for fine-tuning."""
        logger.info(f"Unfreezing last {num_layers} layers")
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True
            
    def get_model_summary(self):
        """Get model summary."""
        return self.model.summary()
        
    def plot_model(self, path=None):
        """Plot model architecture."""
        logger.info("Plotting model architecture")
        if path is None:
            path = OUTPUTS_DIR / "resnet_architecture.png"
        tf.keras.utils.plot_model(
            self.model,
            to_file=str(path),
            show_shapes=True,
            show_layer_names=True
        ) 