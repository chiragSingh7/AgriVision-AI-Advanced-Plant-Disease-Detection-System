import os
import shutil
from pathlib import Path

def clean_directories():
    # Clean notebooks directory
    notebooks_dir = Path("notebooks")
    if notebooks_dir.exists():
        # Remove all files in notebooks directory
        for file in notebooks_dir.glob("*"):
            if file.is_file():
                file.unlink()
        print("Cleaned notebooks directory")

    # Clean outputs directory
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        # Remove all contents of outputs directory
        shutil.rmtree(outputs_dir)
        # Recreate the outputs directory with subdirectories
        outputs_dir.mkdir(exist_ok=True)
        
        # Create necessary subdirectories
        subdirs = [
            "confusion_matrices",
            "classification_reports",
            "performance_plots",
            "disease_analysis",
            "heatmaps",
            "model_checkpoints"
        ]
        for subdir in subdirs:
            (outputs_dir / subdir).mkdir(exist_ok=True)
        print("Cleaned and restructured outputs directory")

def clean_models_directory():
    # Path to models directory
    models_dir = Path("models")
    
    # Remove all files in models directory
    if models_dir.exists():
        for file in models_dir.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"Removed: {file.name}")
        print("\nModels directory cleaned successfully!")
        
        # Create a .gitkeep file to preserve directory in git
        (models_dir / ".gitkeep").touch()
    else:
        models_dir.mkdir(exist_ok=True)
        print("Created new models directory")

if __name__ == "__main__":
    clean_directories()
    clean_models_directory()