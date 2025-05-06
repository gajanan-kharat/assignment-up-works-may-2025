print("Test starting...")
try:
    import torch
    print("Imported torch")
    import cv2
    print("Imported cv2")
    import numpy as np
    print("Imported numpy")
    from ultralytics import YOLO
    print("Imported YOLO")
    print("All imports successful!")
except Exception as e:
    print(f"Error: {str(e)}")
print("Test complete")