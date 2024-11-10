import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('/content/drive/MyDrive/keyboard_detection_model.pt')  # Update path if necessary

# Open the webcam
cap = cv2.VideoCapture(1)  # 0 is typically the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw the detections on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Webcam', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
