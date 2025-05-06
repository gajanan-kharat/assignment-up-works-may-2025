# YOLO Inference Project

This project demonstrates the use of YOLO (You Only Look Once) for object detection and inference. It provides a robust implementation with support for both PyTorch and ONNX runtime, comprehensive logging, and extensive testing capabilities.

## Features

- 🚀 Dual inference support: PyTorch and ONNX runtime
- 📊 Detailed logging system with both file and console output
- 🔄 Automatic model conversion from PyTorch to ONNX format
- 🧪 Comprehensive testing suite
- 🖼️ Image preprocessing and visualization capabilities
- 📈 Confidence score reporting for detections

## Project Structure

- **image.jpeg**: Sample input image for testing
- **pytorch_result.jpg**: Output image with detection visualizations
- **pytorch_test.py**: Quick test script for PyTorch-based inference
- **README.md**: Project documentation
- **requirements.txt**: Project dependencies
- **test_components.py**: Unit tests for individual components
- **test_env.py**: Environment validation script
- **test_minimal.py**: Minimal test suite for quick validation
- **yolo_inference.py**: Main inference script with PyTorch and ONNX support
- **yolo_run.log**: Detailed execution logs
- **yolo11n.pt**: Pre-trained YOLO model weights

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Required packages:
  - torch
  - torchvision
  - ultralytics
  - opencv-python
  - onnx (optional)
  - onnxruntime (optional)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Environment Setup
First, validate your environment:
```bash
python test_env.py
```

### 2. Running Inference
Execute the main inference script:
```bash
python yolo_inference.py
```

The script will:
- Load the YOLO model
- Run inference on the input image
- Convert the model to ONNX format (if supported)
- Run ONNX inference (if available)
- Save results to 'pytorch_result.jpg'

### 3. Testing

#### Quick Validation
```bash
python test_minimal.py
```

#### Component Testing
```bash
python test_components.py
```

#### PyTorch-specific Testing
```bash
python pytorch_test.py
```

## Logging

The project maintains detailed logs in 'yolo_run.log', including:
- Model loading status
- Inference progress
- Error messages and stack traces
- Performance metrics

## Output

- Detection results are saved as 'pytorch_result.jpg'
- Detection details including class IDs and confidence scores are logged
- ONNX model (if conversion is successful) is saved as 'yolo11n.onnx'

## Error Handling

The project includes comprehensive error handling for:
- Missing dependencies
- File not found scenarios
- Model loading failures
- Inference errors
- ONNX conversion issues
