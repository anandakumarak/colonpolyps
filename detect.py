import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv11 segmentation model
model = YOLO("best.pt")  # Change this to your trained model path

# Load video file
video_path = "colonoscopy.mp4"  # Update with your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer (optional: save output video)
out = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Run YOLOv11 model on the frame
    results = model(frame)

    # Draw detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        masks = result.masks.data.cpu().numpy() if result.masks else []  # Masks
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class labels

        for box, mask, score, class_id in zip(boxes, masks, scores, class_ids):
            if score > 0.72:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box[:4])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[class_id]} {score:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw segmentation mask
                mask = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # Show output frame
    cv2.imshow("YOLOv11 Polyp Detection", frame)
    out.write(frame)  # Save frame to output video

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
