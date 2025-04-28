"""
Script to organize plant disease classification data.
"""

import os
import shutil
from pathlib import Path
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# Define paths
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

def create_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def clean_directory(directory):
    """Remove all files and subdirectories in the given directory."""
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
        logger.info(f"Cleaned directory: {directory}")

def organize_data():
    """Organize data into train, validation, and test sets."""
    # Clean existing directories
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        clean_directory(dir_path)

    # Get all disease classes from PlantVillage directory
    source_dir = DATA_DIR / "PlantVillage"
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} not found!")
        return

    disease_classes = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for disease_class in disease_classes:
        logger.info(f"Processing {disease_class.name}")
        
        # Create class directories in train, val, and test
        train_class_dir = TRAIN_DIR / disease_class.name
        val_class_dir = VAL_DIR / disease_class.name
        test_class_dir = TEST_DIR / disease_class.name
        
        for dir_path in [train_class_dir, val_class_dir, test_class_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Get all images in the class
        images = list(disease_class.glob("*.JPG")) + list(disease_class.glob("*.jpg"))
        random.shuffle(images)
        
        # Split images (70% train, 15% val, 15% test)
        n_images = len(images)
        n_train = int(0.7 * n_images)
        n_val = int(0.15 * n_images)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, train_class_dir / img.name)
        
        for img in val_images:
            shutil.copy2(img, val_class_dir / img.name)
        
        for img in test_images:
            shutil.copy2(img, test_class_dir / img.name)
        
        logger.info(f"Split {disease_class.name}: {len(train_images)} train, "
                   f"{len(val_images)} val, {len(test_images)} test")

def cleanup_unnecessary_files():
    """Remove unnecessary files and directories."""
    # Remove .DS_Store files
    for ds_store in DATA_DIR.rglob(".DS_Store"):
        ds_store.unlink()
        logger.info(f"Removed: {ds_store}")
    
    # Remove __MACOSX directory
    macosx_dir = DATA_DIR / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
        logger.info(f"Removed: {macosx_dir}")
    
    # Remove processed directory
    processed_dir = DATA_DIR / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        logger.info(f"Removed: {processed_dir}")

def print_data_summary():
    """Print summary of the organized data."""
    logger.info("\nData Organization Summary:")
    logger.info("=" * 50)
    
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        total_images = 0
        logger.info(f"\n{split_dir.name.upper()} Set:")
        logger.info("-" * 20)
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                num_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.JPG")))
                total_images += num_images
                logger.info(f"{class_dir.name}: {num_images} images")
        
        logger.info(f"Total {split_dir.name} images: {total_images}")

def create_directory_structure():
    """Create required directory structure."""
    directories = [
        'data/raw/bacterial_blight',
        'data/raw/blast',
        'data/raw/brown_spot',
        'data/raw/healthy',
        'data/processed',
        'outputs/logs',
        'outputs/models',
        'outputs/visualizations',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    logger.info("Starting data organization")
    create_directories()
    organize_data()
    cleanup_unnecessary_files()
    print_data_summary()
    create_directory_structure()
    logger.info("Data organization completed")
