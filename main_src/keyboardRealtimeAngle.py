import torch
from ultralytics import YOLO  # Import YOLO library
import cv2
import numpy as np
from collections import deque

# Check if MPS is available (for macOS) and set it as the device, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the YOLO model
model = YOLO('keyboard_detection_model.pt')   # Replace with the path to your trained model

# Open the video capture (0 for the default camera)
cap = cv2.VideoCapture(1)

# Rolling average setup for angle smoothing
angle_buffer = deque(maxlen=5)  # Store the last 5 angle readings

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO object detection on the current frame
    results = model.predict(frame, device=device)

    # Iterate through the detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates as integers
            confidence = box.conf[0]  # Confidence score

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Crop the detected object region for angle estimation
            cropped_object = frame[y1:y2, x1:x2]

            # Convert to grayscale
            gray = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Get the minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[-1]  # Angle of orientation

                # Correct the angle if necessary
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90

                # Add the angle to the rolling buffer
                angle_buffer.append(angle)

                # Calculate the rolling average if buffer has enough readings
                smoothed_angle = np.mean(angle_buffer) if len(angle_buffer) > 1 else angle

                # Display the smoothed angle on the frame
                cv2.putText(frame, f"Angle: {smoothed_angle:.2f} degrees", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display guidance messages for the user with specific rotation instructions
                if abs(smoothed_angle) < 1.5:  # Adjusted threshold for very strict alignment
                    cv2.putText(frame, "Aligned! Move straight.", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif smoothed_angle > 1.5:
                    cv2.putText(frame, f"Rotate clockwise by {smoothed_angle:.2f} degrees", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif smoothed_angle < -1.5:
                    cv2.putText(frame, f"Rotate counterclockwise by {abs(smoothed_angle):.2f} degrees", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Detection and Orientation", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
