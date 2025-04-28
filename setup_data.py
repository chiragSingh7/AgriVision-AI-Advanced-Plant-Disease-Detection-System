import os
from pathlib import Path
import shutil

def setup_data_directories():
    """Create and setup data directories."""
    # Create main directories
    directories = [
        'data/raw/bacterial_blight',
        'data/raw/blast',
        'data/raw/brown_spot',
        'data/raw/healthy',
        'data/processed',
        'outputs/logs',
        'outputs/models',
        'outputs/visualizations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def verify_data():
    """Verify data presence in directories."""
    classes = ['bacterial_blight', 'blast', 'brown_spot', 'healthy']
    missing_data = []
    
    for class_name in classes:
        dir_path = Path('data/raw') / class_name
        if not dir_path.exists() or not any(dir_path.glob('*.jpg')):
            missing_data.append(class_name)
    
    if missing_data:
        print("\nMissing data for classes:")
        for class_name in missing_data:
            print(f"- {class_name}")
        print("\nPlease add image data to these directories.")
        print("Images should be in .jpg format")
        print("Directory structure should be:")
        print("data/raw/")
        for class_name in classes:
            print(f"  └── {class_name}/")
            print("       └── image1.jpg")
            print("       └── image2.jpg")
            print("       └── ...")

if __name__ == "__main__":
    setup_data_directories()
    verify_data()
