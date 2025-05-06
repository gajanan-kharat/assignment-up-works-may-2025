import sys
import cv2
from ultralytics import YOLO

print("Starting PyTorch inference test...")

try:
    # Load the model
    print("Loading YOLO model...")
    model = YOLO("yolo11n.pt")
    print("Model loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    results = model("image.jpeg")
    print("Inference completed!")

    # Save and display results
    print("\nProcessing results...")
    for r in results:
        im_array = r.plot()
        cv2.imwrite('pytorch_result.jpg', im_array)
        
        print("\nDetection Results:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"- Class {cls} detected with {conf:.2%} confidence")

    print("\nResults saved to pytorch_result.jpg")
    print("Test completed successfully!")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)