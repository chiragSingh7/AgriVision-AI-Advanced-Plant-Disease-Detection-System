"""
Enhanced heatmap generation module for plant disease classification.
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import logging
from pathlib import Path
from src.config import VIS_CONFIG

class HeatmapGenerator:
    """Enhanced heatmap generator class for model interpretation."""
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize heatmap generator with model.
        
        Args:
            model: Trained model for generating heatmaps
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.last_conv_layer = self._find_last_conv_layer()
        
        # Visualization settings from config
        self.fig_size = VIS_CONFIG.get("figure_size", (12, 8))
        self.dpi = VIS_CONFIG.get("dpi", 100)
        self.output_dir = Path(VIS_CONFIG.get("output_directory", "outputs/visualizations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized HeatmapGenerator")
        if self.last_conv_layer:
            self.logger.info(f"Last convolutional layer: {self.last_conv_layer.name}")
    
    def _find_last_conv_layer(self) -> Optional[tf.keras.layers.Layer]:
        """
        Find the last convolutional layer in the model.
        
        Returns:
            Last convolutional layer or None if not found
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        
        self.logger.warning("No convolutional layer found in the model")
        return None
    
    def generate_grad_cam(self, img: np.ndarray, 
                         class_idx: Optional[int] = None,
                         layer_name: Optional[str] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            img: Input image (should be preprocessed)
            class_idx: Target class index (uses predicted class if None)
            layer_name: Name of the layer to use for Grad-CAM (uses last conv if None)
            
        Returns:
            Generated heatmap as numpy array
        """
        try:
            # Get target layer
            target_layer = self.model.get_layer(layer_name) if layer_name else self.last_conv_layer
            if target_layer is None:
                raise ValueError("Target layer not found")
            
            # Create gradient model
            grad_model = tf.keras.Model(
                inputs=[self.model.inputs],
                outputs=[target_layer.output, self.model.output]
            )
            
            # Get model predictions
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(img)
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                class_output = predictions[:, class_idx]
            
            # Calculate gradients
            grads = tape.gradient(class_output, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            heatmap = tf.reduce_sum(
                tf.multiply(pooled_grads, conv_output), axis=-1
            )
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            self.logger.error(f"Error generating Grad-CAM: {str(e)}")
            return np.zeros((img.shape[0], img.shape[1]))
    
    def visualize_heatmap(self, img: np.ndarray, heatmap: np.ndarray,
                         alpha: float = 0.4, 
                         save_path: Optional[Union[str, Path]] = None,
                         show_original: bool = True) -> np.ndarray:
        """
        Overlay heatmap on the original image with enhanced visualization.
        
        Args:
            img: Original image
            heatmap: Generated heatmap
            alpha: Transparency of the heatmap overlay
            save_path: Path to save the visualization
            show_original: Whether to include original image in visualization
            
        Returns:
            Visualization with heatmap overlay
        """
        try:
            # Ensure proper image format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            # Resize heatmap to match image size
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Ensure image is RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # Handle RGBA images
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Create overlay
            overlay = cv2.addWeighted(img, 1-alpha, colored_heatmap, alpha, 0)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=self.fig_size)
                
                if show_original:
                    plt.subplot(131)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.imshow(heatmap, cmap='jet')
                    plt.title('Heatmap')
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    plt.title('Overlay')
                    plt.axis('off')
                else:
                    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    plt.title('Heatmap Overlay')
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Saved heatmap visualization to {save_path}")
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"Error visualizing heatmap: {str(e)}")
            return img
    
    def generate_multiple_heatmaps(self, images: np.ndarray, 
                                 class_indices: Optional[list] = None,
                                 save_dir: Optional[Union[str, Path]] = None,
                                 batch_size: int = 32) -> list:
        """
        Generate heatmaps for multiple images with batch processing.
        
        Args:
            images: Array of input images
            class_indices: List of target class indices
            save_dir: Directory to save visualizations
            batch_size: Batch size for processing
            
        Returns:
            List of generated heatmaps
        """
        heatmaps = []
        save_dir = Path(save_dir) if save_dir else None
        
        try:
            # Process images in batches
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_indices = class_indices[i:i + batch_size] if class_indices else None
                
                # Generate heatmaps for batch
                for j, img in enumerate(batch_images):
                    class_idx = batch_indices[j] if batch_indices else None
                    heatmap = self.generate_grad_cam(
                        img[np.newaxis, ...], 
                        class_idx=class_idx
                    )
                    heatmaps.append(heatmap)
                    
                    if save_dir:
                        save_path = save_dir / f"heatmap_{i+j}.png"
                        self.visualize_heatmap(img, heatmap, save_path=save_path)
                
                self.logger.info(f"Processed batch {i//batch_size + 1}")
            
            return heatmaps
            
        except Exception as e:
            self.logger.error(f"Error generating multiple heatmaps: {str(e)}")
            return []
    
    def analyze_activation_patterns(self, img: np.ndarray, 
                                 layer_names: Optional[list] = None) -> dict:
        """
        Analyze activation patterns in convolutional layers.
        
        Args:
            img: Input image
            layer_names: List of layer names to analyze (all conv layers if None)
            
        Returns:
            Dictionary of activation statistics
        """
        try:
            # Get target layers
            if layer_names:
                target_layers = [
                    layer for layer in self.model.layers 
                    if layer.name in layer_names
                ]
            else:
                target_layers = [
                    layer for layer in self.model.layers 
                    if isinstance(layer, tf.keras.layers.Conv2D)
                ]
            
            # Create activation model
            activation_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[layer.output for layer in target_layers]
            )
            
            # Get activations
            activations = activation_model.predict(img[np.newaxis, ...])
            if not isinstance(activations, list):
                activations = [activations]
            
            # Calculate statistics
            stats = {}
            for layer, activation in zip(target_layers, activations):
                stats[layer.name] = {
                    'mean': float(np.mean(activation)),
                    'std': float(np.std(activation)),
                    'max': float(np.max(activation)),
                    'min': float(np.min(activation)),
                    'shape': activation.shape,
                    'active_filters': int(np.sum(np.max(activation, axis=(0,1)) > 0)),
                    'total_filters': activation.shape[-1]
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing activation patterns: {str(e)}")
            return {}
