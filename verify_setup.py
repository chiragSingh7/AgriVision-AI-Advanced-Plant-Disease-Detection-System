#!/usr/bin/env python3

import tensorflow as tf
import sys
import os
from pathlib import Path

def verify_setup():
    """Verify the setup of the plant disease classification system."""
    print("\n=== Plant Disease Classification System Verification ===\n")
    
    # 1. Check Python and TensorFlow
    print("System Information:")
    print("-----------------")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # 2. Check GPU/Metal support
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU devices available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # 3. Check directory structure
    print("\nDirectory Structure:")
    print("-------------------")
    directories = [
        'data/raw',
        'data/processed',
        'outputs/logs',
        'outputs/models',
        'outputs/visualizations'
    ]
    
    all_exist = True
    for dir_path in directories:
        path = Path(dir_path)
        exists = path.exists()
        print(f"{dir_path}: {'✓' if exists else '✗'}")
        if not exists:
            all_exist = False
            path.mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {dir_path}")
    
    # 4. Check write permissions
    print("\nPermission Check:")
    print("----------------")
    try:
        test_file = Path('outputs/logs/test.txt')
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("Test write permission")
        test_file.unlink()  # Remove test file
        print("Write permissions: ✓")
    except Exception as e:
        print(f"Write permissions: ✗ ({str(e)})")
    
    print("\nVerification Complete!")
    print("=====================")
    if all_exist:
        print("Status: Ready to run")
    else:
        print("Status: Directories created, ready to run")

if __name__ == "__main__":
    verify_setup()
