"""
Enhanced data processing module for plant disease classification.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
from src.config import DATA_CONFIG
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers

class DataProcessor:
    """Enhanced data processor class with detailed data handling and augmentation."""
    
    def __init__(self):
        """Initialize data processor with configuration settings."""
        self.logger = logging.getLogger(__name__)
        self.img_size = tuple(DATA_CONFIG["image_size"])
        self.batch_size = DATA_CONFIG["batch_size"]
        self.class_names = DATA_CONFIG["class_names"]
        self.validation_split = DATA_CONFIG["validation_split"]
        self.test_split = DATA_CONFIG["test_split"]
        self.seed = DATA_CONFIG["random_seed"]
        
        # Data augmentation parameters
        self.augmentation_params = DATA_CONFIG["augmentation"]
        
        self.logger.info("Initialized DataProcessor with configurations:")
        self.logger.info(f"Image size: {self.img_size}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of classes: {len(self.class_names)}")
        self.logger.info(f"Augmentation params: {self.augmentation_params}")
        
        self.augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.2),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1)
        ])
    
    def load_and_preprocess_data(self, data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and preprocess the dataset with detailed progress tracking.
        
        Args:
            data_dir: Directory containing the dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        self.logger.info(f"Loading data from {data_dir}")
        data_dir = Path(data_dir)
        
        # Load all image paths and labels
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(tqdm(self.class_names, desc="Loading classes")):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            image_paths, labels,
            test_size=self.test_split,
            stratify=labels,
            random_state=self.seed
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.validation_split,
            stratify=y_train_val,
            random_state=self.seed
        )
        
        # Create datasets
        train_dataset = self._create_dataset(X_train, y_train, is_training=True)
        val_dataset = self._create_dataset(X_val, y_val, is_training=False)
        test_dataset = self._create_dataset(X_test, y_test, is_training=False)
        
        # Log dataset statistics
        self.logger.info("\nDataset Statistics:")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Validation samples: {len(X_val)}")
        self.logger.info(f"Test samples: {len(X_test)}")
        
        class_distribution = self._get_class_distribution(labels)
        self.logger.info("\nClass Distribution:")
        for class_name, count in class_distribution.items():
            self.logger.info(f"{class_name}: {count}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_dataset(self, image_paths: np.ndarray, labels: np.ndarray, 
                       is_training: bool = False) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset with proper preprocessing and augmentation.
        
        Args:
            image_paths: Array of image paths
            labels: Array of labels
            is_training: Whether this is a training dataset
            
        Returns:
            TensorFlow dataset
        """
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Add loading and preprocessing
        dataset = dataset.map(
            lambda x, y: (self._load_and_preprocess_image(x, is_training), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle if training
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths), seed=self.seed)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def _load_and_preprocess_image(self, image_path: tf.Tensor, 
                                 is_training: bool = False) -> tf.Tensor:
        """
        Load and preprocess a single image with augmentation during training.
        
        Args:
            image_path: Path to the image
            is_training: Whether this is for training
            
        Returns:
            Preprocessed image tensor
        """
        # Read image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        
        # Resize
        image = tf.image.resize(image, self.img_size)
        
        # Augmentation during training
        if is_training:
            if self.augmentation_params["random_flip"]:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
            
            if self.augmentation_params["random_rotation"]:
                image = tf.image.rot90(
                    image,
                    k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
                )
            
            if self.augmentation_params["random_brightness"]:
                image = tf.image.random_brightness(image, 0.2)
            
            if self.augmentation_params["random_contrast"]:
                image = tf.image.random_contrast(image, 0.8, 1.2)
            
            if self.augmentation_params["random_saturation"]:
                image = tf.image.random_saturation(image, 0.8, 1.2)
            
            if self.augmentation_params["random_hue"]:
                image = tf.image.random_hue(image, 0.1)
        
        # Normalize
        image = image / 255.0
        
        return image
    
    def _get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get the distribution of classes in the dataset.
        
        Args:
            labels: Array of labels
            
        Returns:
            Dictionary mapping class names to counts
        """
        unique, counts = np.unique(labels, return_counts=True)
        return {self.class_names[i]: count for i, count in zip(unique, counts)}
    
    def create_data_generator(self, data_dir, is_training=False):
        """Create data generator with augmentation for training."""
        try:
            if is_training:
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=self.augmentation_params["rotation_range"],
                    width_shift_range=self.augmentation_params["width_shift_range"],
                    height_shift_range=self.augmentation_params["height_shift_range"],
                    horizontal_flip=self.augmentation_params["horizontal_flip"],
                    vertical_flip=self.augmentation_params["vertical_flip"],
                    zoom_range=self.augmentation_params["zoom_range"],
                    shear_range=self.augmentation_params["shear_range"],
                    fill_mode=self.augmentation_params["fill_mode"],
                    brightness_range=self.augmentation_params["brightness_range"]
                )
            else:
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255
                )
            
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=is_training
            )
            
            self.logger.info(f"Created data generator for {'training' if is_training else 'evaluation'}")
            return generator
            
        except Exception as e:
            self.logger.error(f"Error creating data generator: {str(e)}")
            raise
    
    def load_sample_data(self, n_samples=5):
        """Load a small sample of data for validation."""
        try:
            sample_data = []
            for class_dir in Path('data/raw').iterdir():
                if class_dir.is_dir():
                    files = list(class_dir.glob('*.jpg'))[:n_samples]
                    for file in files:
                        img = tf.keras.preprocessing.image.load_img(
                            file, 
                            target_size=(224, 224)
                        )
                        sample_data.append(
                            tf.keras.preprocessing.image.img_to_array(img)
                        )
            return np.array(sample_data) if sample_data else None
            
        except Exception as e:
            self.logger.error(f"Error loading sample data: {str(e)}")
            raise
    
    def save_preprocessed_data(self, dataset: tf.data.Dataset, 
                             save_dir: str) -> None:
        """
        Save preprocessed data to disk for faster loading.
        
        Args:
            dataset: Dataset to save
            save_dir: Directory to save the data
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (images, labels) in enumerate(tqdm(dataset, desc="Saving data")):
            batch_dir = save_dir / f"batch_{i}"
            batch_dir.mkdir(exist_ok=True)
            
            for j, (image, label) in enumerate(zip(images, labels)):
                image_path = batch_dir / f"image_{j}.npy"
                label_path = batch_dir / f"label_{j}.npy"
                
                np.save(image_path, image.numpy())
                np.save(label_path, label.numpy())
    
    def load_preprocessed_data(self, data_dir: str) -> tf.data.Dataset:
        """
        Load preprocessed data from disk.
        
        Args:
            data_dir: Directory containing preprocessed data
            
        Returns:
            TensorFlow dataset
        """
        data_dir = Path(data_dir)
        
        # Get all batch directories
        batch_dirs = sorted(data_dir.glob("batch_*"))
        
        images = []
        labels = []
        
        for batch_dir in tqdm(batch_dirs, desc="Loading preprocessed data"):
            image_files = sorted(batch_dir.glob("image_*.npy"))
            label_files = sorted(batch_dir.glob("label_*.npy"))
            
            for image_file, label_file in zip(image_files, label_files):
                images.append(np.load(image_file))
                labels.append(np.load(label_file))
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset 