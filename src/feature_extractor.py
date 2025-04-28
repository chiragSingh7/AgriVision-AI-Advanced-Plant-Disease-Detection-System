"""
Enhanced feature extractor module for plant disease classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from src.config import FEATURE_CONFIG

class FeatureExtractor:
    """Enhanced feature extractor class with multiple extraction methods."""
    
    def __init__(self):
        """Initialize feature extractor with configuration settings."""
        self.logger = logging.getLogger(__name__)
        
        # Get configuration parameters with defaults if not specified
        self.img_size = tuple(FEATURE_CONFIG.get("image_size", (224, 224)))
        self.batch_size = FEATURE_CONFIG.get("batch_size", 32)
        
        # Initialize feature extraction flags
        self.use_color = FEATURE_CONFIG.get("color_features", True)
        self.use_texture = FEATURE_CONFIG.get("texture_features", True)
        self.use_shape = FEATURE_CONFIG.get("shape_features", True)
        self.use_hog = FEATURE_CONFIG.get("hog_features", True)
        self.use_lbp = FEATURE_CONFIG.get("lbp_features", True)
        self.use_deep = FEATURE_CONFIG.get("deep_features", True)
        
        # Add PCA configuration
        self.use_pca = FEATURE_CONFIG.get("use_pca", False)
        self.n_components = FEATURE_CONFIG.get("pca_components", 100)
        
        # Initialize feature extractors
        self._init_feature_extractors()
        
        # Log configuration
        self.logger.info("Initialized FeatureExtractor with configurations:")
        self.logger.info(f"Image size: {self.img_size}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Feature types enabled:")
        self.logger.info(f"  - Color features: {self.use_color}")
        self.logger.info(f"  - Texture features: {self.use_texture}")
        self.logger.info(f"  - Shape features: {self.use_shape}")
        self.logger.info(f"  - HOG features: {self.use_hog}")
        self.logger.info(f"  - LBP features: {self.use_lbp}")
        self.logger.info(f"  - Deep features: {self.use_deep}")
        self.logger.info(f"  - PCA enabled: {self.use_pca}")
        if self.use_pca:
            self.logger.info(f"  - PCA components: {self.n_components}")
    
    def _init_feature_extractors(self):
        """Initialize various feature extraction models."""
        try:
            # ResNet feature extractor
            if self.use_deep:
                self.resnet = ResNet50V2(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
            
            # Initialize PCA if enabled
            self.pca = None
            if self.use_pca:
                self.pca = PCA(n_components=self.n_components)
            
            self.logger.info("Feature extraction models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature extractors: {str(e)}")
            raise
    
    def extract_deep_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract deep features using ResNet with progress tracking.
        
        Args:
            images: Array of images
            
        Returns:
            Array of deep features
        """
        self.logger.info("Extracting deep features...")
        
        # Preprocess images
        preprocessed_images = preprocess_input(images)
        
        # Extract features in batches
        features = []
        n_batches = int(np.ceil(len(images) / self.batch_size))
        
        for i in tqdm(range(n_batches), desc="Extracting deep features"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(images))
            batch = preprocessed_images[start_idx:end_idx]
            
            batch_features = self.resnet.predict(batch, verbose=0)
            features.append(batch_features)
        
        features = np.concatenate(features, axis=0)
        self.logger.info(f"Extracted deep features with shape: {features.shape}")
        
        return features
    
    def extract_color_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract color features (histograms) from images.
        
        Args:
            images: Array of images
            
        Returns:
            Array of color features
        """
        self.logger.info("Extracting color features...")
        
        color_features = []
        for image in tqdm(images, desc="Extracting color features"):
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Calculate histograms
            hist_rgb = self._calculate_color_histogram(image)
            hist_hsv = self._calculate_color_histogram(hsv)
            hist_lab = self._calculate_color_histogram(lab)
            
            # Combine features
            color_feature = np.concatenate([hist_rgb, hist_hsv, hist_lab])
            color_features.append(color_feature)
        
        color_features = np.array(color_features)
        self.logger.info(f"Extracted color features with shape: {color_features.shape}")
        
        return color_features
    
    def extract_texture_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract texture features using various methods.
        
        Args:
            images: Array of images
            
        Returns:
            Array of texture features
        """
        self.logger.info("Extracting texture features...")
        
        texture_features = []
        for image in tqdm(images, desc="Extracting texture features"):
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Calculate GLCM features
            glcm_features = self._calculate_glcm_features(gray)
            
            # Calculate LBP features
            lbp_features = self._calculate_lbp_features(gray)
            
            # Combine features
            texture_feature = np.concatenate([glcm_features, lbp_features])
            texture_features.append(texture_feature)
        
        texture_features = np.array(texture_features)
        self.logger.info(f"Extracted texture features with shape: {texture_features.shape}")
        
        return texture_features
    
    def extract_shape_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract shape features from segmented disease regions.
        
        Args:
            images: Array of images
            
        Returns:
            Array of shape features
        """
        self.logger.info("Extracting shape features...")
        
        shape_features = []
        for image in tqdm(images, desc="Extracting shape features"):
            # Segment disease region
            mask = self.segment_disease(image)
            
            # Calculate shape features
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Calculate various shape metrics
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate Hu moments
                moments = cv2.moments(largest_contour)
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Combine shape features
                shape_feature = np.concatenate([[area, perimeter, circularity], hu_moments])
            else:
                shape_feature = np.zeros(10)  # Default features if no contour found
            
            shape_features.append(shape_feature)
        
        shape_features = np.array(shape_features)
        self.logger.info(f"Extracted shape features with shape: {shape_features.shape}")
        
        return shape_features
    
    def extract_all_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract all features and combine them.
        
        Args:
            images: Array of images
            
        Returns:
            Array of combined features
        """
        self.logger.info("Extracting all features...")
        
        # Extract different types of features
        deep_features = self.extract_deep_features(images)
        color_features = self.extract_color_features(images)
        texture_features = self.extract_texture_features(images)
        shape_features = self.extract_shape_features(images)
        
        # Combine all features
        combined_features = np.concatenate([
            deep_features,
            color_features,
            texture_features,
            shape_features
        ], axis=1)
        
        # Apply PCA if configured
        if self.use_pca:
            if self.pca is None or not hasattr(self.pca, 'components_'):
                self.logger.info("Fitting PCA...")
                combined_features = self.pca.fit_transform(combined_features)
            else:
                self.logger.info("Applying pre-fitted PCA...")
                combined_features = self.pca.transform(combined_features)
        
        self.logger.info(f"Final feature shape: {combined_features.shape}")
        return combined_features
    
    def _calculate_color_histogram(self, image: np.ndarray, 
                                 bins: int = 32) -> np.ndarray:
        """
        Calculate color histogram for an image.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            
        Returns:
            Flattened color histogram
        """
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins] * 3, 
                           [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def _calculate_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate GLCM (Gray-Level Co-occurrence Matrix) features.
        
        Args:
            image: Grayscale input image
            
        Returns:
            GLCM features
        """
        glcm = np.zeros((16, 16))
        h, w = image.shape
        normalized_img = (image / 16).astype(np.uint8)
        
        for i in range(h-1):
            for j in range(w-1):
                glcm[normalized_img[i, j], normalized_img[i+1, j+1]] += 1
        
        glcm = glcm / glcm.sum()
        
        # Calculate GLCM properties
        contrast = np.sum(np.square(np.arange(16)[:, None] - np.arange(16)) * glcm)
        correlation = np.sum((np.arange(16)[:, None] - np.mean(glcm)) * 
                           (np.arange(16) - np.mean(glcm)) * glcm) / (np.std(glcm) ** 2)
        energy = np.sum(np.square(glcm))
        homogeneity = np.sum(glcm / (1 + np.square(np.arange(16)[:, None] - np.arange(16))))
        
        return np.array([contrast, correlation, energy, homogeneity])
    
    def _calculate_lbp_features(self, image: np.ndarray, 
                              points: int = 8, radius: int = 1) -> np.ndarray:
        """
        Calculate Local Binary Pattern features.
        
        Args:
            image: Grayscale input image
            points: Number of points in LBP calculation
            radius: Radius for LBP calculation
            
        Returns:
            LBP features
        """
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary = image[i-radius:i+radius+1, j-radius:j+radius+1].flatten() >= center
                lbp[i, j] = sum(binary * (2 ** np.arange(len(binary))))
        
        hist, _ = np.histogram(lbp, bins=points, range=(0, 2**points))
        hist = hist.astype(float) / hist.sum()
        return hist
    
    def segment_disease(self, image: np.ndarray) -> np.ndarray:
        """
        Segment disease regions in the image.
        
        Args:
            image: Input image
            
        Returns:
            Binary mask of segmented regions
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
        # Convert to HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        
        # Create mask for disease regions
        lower = np.array([20, 20, 20])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def save_features(self, features: np.ndarray, save_path: str) -> None:
        """
        Save extracted features to disk.
        
        Args:
            features: Array of features
            save_path: Path to save features
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, features)
        self.logger.info(f"Saved features to {save_path}")
    
    def load_features(self, load_path: str) -> np.ndarray:
        """
        Load features from disk.
        
        Args:
            load_path: Path to load features from
            
        Returns:
            Array of features
        """
        features = np.load(load_path)
        self.logger.info(f"Loaded features with shape: {features.shape}")
        return features
