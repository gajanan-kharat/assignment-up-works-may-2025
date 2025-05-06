import os
import sys
import logging

# Configure logging to both file and console with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('component_test.log', mode='w')
    ]
)

def test_imports():
    """Test importing required packages"""
    logging.info("Testing imports...")
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        
        import cv2
        logging.info(f"OpenCV version: {cv2.__version__}")
        
        import numpy as np
        logging.info(f"NumPy version: {np.__version__}")
        
        from ultralytics import YOLO
        logging.info("YOLO imported successfully")
        
        import onnxruntime as ort
        logging.info(f"ONNX Runtime version: {ort.__version__}")
        
        return True
    except Exception as e:
        logging.error(f"Import error: {str(e)}")
        return False

def test_model_loading():
    """Test loading the YOLO model"""
    logging.info("Testing model loading...")
    try:
        if not os.path.exists("yolo11n.pt"):
            logging.error("Model file yolo11n.pt not found!")
            return False
            
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        logging.info("Model loaded successfully!")
        return True
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        return False

def test_image_loading():
    """Test loading and preprocessing the image"""
    logging.info("Testing image loading...")
    try:
        if not os.path.exists("image.jpeg"):
            logging.error("Image file image.jpeg not found!")
            return False
            
        import cv2
        img = cv2.imread("image.jpeg")
        if img is None:
            logging.error("Failed to load image!")
            return False
            
        logging.info(f"Image loaded successfully! Shape: {img.shape}")
        return True
    except Exception as e:
        logging.error(f"Image loading error: {str(e)}")
        return False

if __name__ == "__main__":
    logging.info("=== Starting component tests ===")
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Image Loading Test", test_image_loading)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logging.info(f"\nRunning {test_name}...")
        try:
            result = test_func()
            if result:
                logging.info(f"{test_name} PASSED")
            else:
                logging.error(f"{test_name} FAILED")
                all_passed = False
        except Exception as e:
            logging.error(f"{test_name} FAILED with exception: {str(e)}")
            all_passed = False
    
    if all_passed:
        logging.info("\n=== All component tests passed! ===")
    else:
        logging.error("\n=== Some tests failed! Check the log for details ===")
        sys.exit(1)
