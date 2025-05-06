import sys
import os

# Redirect stdout to a file
with open('test_output.txt', 'w') as f:
    # Store the original stdout
    original_stdout = sys.stdout
    sys.stdout = f
    
    try:
        print("Starting environment test...")
        
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        from ultralytics import YOLO
        print("Successfully imported YOLO")
        
        print("\nChecking for model file...")
        if os.path.exists("yolo11n.pt"):
            print("Found yolo11n.pt")