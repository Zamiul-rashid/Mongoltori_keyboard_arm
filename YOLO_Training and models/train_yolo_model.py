import torch
from ultralytics import YOLO  # Import your YOLO library

# Check if MPS is available and set it as the device
device = torch.device("cuda") if torch.backends.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the YOLO model
model = YOLO("yolov8n.pt")  # Load your YOLO model
#preferbly use the large model if available

# Train the model and specify the device
model.train(data="###location to yml file####", epochs=50, imgsz=640, device=device)

# Save the trained model
model.save(r"YOLO_Training and models/pt_files")  # Save the model to a specified file path
