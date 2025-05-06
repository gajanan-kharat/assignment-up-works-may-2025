# YOLO Inference Project

## Overview
This project demonstrates the use of YOLO (You Only Look Once) for object detection. It includes scripts for running inference using PyTorch and ONNX models, converting models to ONNX format, and verifying required files.

## Requirements
To run this project, ensure you have the following installed:

- Python 3.8 or later
- Required Python packages (install using `pip install -r requirements.txt`):
  - torch
  - torchvision
  - ultralytics
  - onnx
  - onnxruntime
  - opencv-python
  - numpy

## Files in the Project
- `yolo_inference.py`: Main script for running inference and model conversion.
- `yolo11n.pt`: YOLO model file.
- `requirements.txt`: List of required Python packages.
- `pytorch_result.jpg`: Output image after PyTorch inference.
- `yolo_run.log`: Log file for the inference process.

## How to Run
1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the input image in the project directory and update the `IMAGE_NAME` variable in `yolo_inference.py` if needed.

3. Run the main script:
   ```bash
   python yolo_inference.py
   ```

4. Check the output image (`pytorch_result.jpg`) and logs (`yolo_run.log`) for results.

## Notes
- Ensure the input image file exists and is correctly named as per the `IMAGE_NAME` variable in the script.
- The ONNX conversion step requires ONNX and ONNX Runtime to be installed.
- Logs provide detailed information about the process and any errors encountered.

## Troubleshooting
- If you encounter errors, check the `yolo_run.log` file for details.
- Ensure all required files are present in the project directory.

## Enhancements

### Additional Features
- **ONNX Conversion**: The project supports converting PyTorch models to ONNX format for broader compatibility.
- **Logging**: Detailed logs are generated in `yolo_run.log` to help debug and monitor the process.
- **Error Handling**: The script includes robust error handling to ensure smooth execution.

### Future Improvements
- **Batch Processing**: Add support for processing multiple images in a single run.
- **Custom Models**: Allow users to specify custom YOLO models via command-line arguments.
- **Visualization**: Integrate a visualization tool to display inference results interactively.
- **Docker Support**: Provide a Dockerfile for containerized execution.

