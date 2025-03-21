from ultralytics import YOLO

if __name__ == "__main__":  # Ensure proper multiprocessing handling
    model_path = "yolo11x-seg.pt"  # Make sure this file exists
    try:
        model = YOLO(model_path)  # Load the YOLOv11 segmentation model
        print("Model loaded successfully!")

        # Train on your custom dataset
        model.train(
            data="D:/Projects/colon_cancer/data.yaml",  # Update with your dataset config
            epochs=50,
            batch=1,
            imgsz=640,
            device="cuda"  # Use GPU if available
        )

    except Exception as e:
        print(f"Error loading model: {e}")
