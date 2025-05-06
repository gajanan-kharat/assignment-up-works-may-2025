import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    logging.info("Starting imports...")
    import torch
    from ultralytics import YOLO
    import cv2
    import numpy as np
    logging.info("Base packages imported successfully")
    
    # Try importing onnx related packages
    try:
        import onnx
        import onnxruntime as ort
        ONNX_AVAILABLE = True
        logging.info("ONNX support is available")
    except ImportError:
        ONNX_AVAILABLE = False
        logging.warning("ONNX support not available - will use PyTorch only")
        
except ImportError as e:
    logging.error(f"Error importing required packages: {e}")
    logging.error("Please make sure to install all requirements using:")
    logging.error("pip install torch torchvision ultralytics onnx onnxruntime opencv-python")
    sys.exit(1)

# Define the image name once
IMAGE_NAME = 'image.jpeg'

# Update the image_path to use the variable
image_path = IMAGE_NAME

def verify_files_exist():
    """Verify that all required files exist"""
    required_files = {
        'YOLO model': 'yolo11n.pt',
        'Input image': IMAGE_NAME
    }
    
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logging.info(f"Successfully loaded and preprocessed image: {image_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return False

def run_pytorch_inference():
    """Run inference using PyTorch model"""
    try:
        logging.info("Loading PyTorch model...")
        model = YOLO("yolo11n.pt")
        
        logging.info("Running PyTorch inference...")
        results = model(IMAGE_NAME)
        
        # Save results
        for r in results:
            im_array = r.plot()
            cv2.imwrite('pytorch_result.jpg', im_array)
        
        logging.info("PyTorch results saved to pytorch_result.jpg")
        return results
    except Exception as e:
        logging.error(f"Error during PyTorch inference: {e}")
        sys.exit(1)

def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    try:
        logging.info("Converting model to ONNX...")
        model = YOLO("yolo11n.pt")
        success = model.export(format="onnx")
        if success:
            logging.info("ONNX conversion complete")
        else:
            logging.error("ONNX conversion failed")
        return success
    except Exception as e:
        logging.error(f"Error during ONNX conversion: {e}")
        sys.exit(1)

def run_onnx_inference():
    """Run inference using ONNX model"""
    try:
        logging.info("Loading ONNX model...")
        session = ort.InferenceSession("yolo11n.onnx", providers=['CPUExecutionProvider'])
        
        img = load_and_preprocess_image(IMAGE_NAME)
        input_name = session.get_inputs()[0].name
        input_data = np.expand_dims(img, 0)
        
        logging.info("Running ONNX inference...")
        outputs = session.run(None, {input_name: input_data})
        logging.info("ONNX inference complete")
        return outputs
    except Exception as e:
        logging.error(f"Error during ONNX inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        logging.info("=== Starting AI.SEE assessment tasks ===")
        
        # Verify files exist
        if not verify_files_exist():
            sys.exit(1)
        
        logging.info("Step 1: PyTorch Inference")
        pytorch_results = run_pytorch_inference()
        
        logging.info("Step 2: Convert to ONNX")
        convert_to_onnx()
        
        logging.info("Step 3: ONNX Inference")
        onnx_results = run_onnx_inference()
        
        logging.info("=== All tasks completed successfully! ===")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)