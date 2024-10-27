import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import json
from PIL import Image

class KeyboardOCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            det_db_thresh=0.3,  # Lower threshold for detection
            det_db_box_thresh=0.3,  # Lower box threshold
            det_db_unclip_ratio=1.6,  # Adjusted unclip ratio for smaller text
            rec_char_dict_path='your_custom_dict.txt'  # Path to custom dictionary
        )

    def preprocess_image(self, image):
        """
        Enhance image for better key detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Edge enhancement
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened

    def detect_keys(self, image_path):
        """
        Detect keyboard keys with enhanced preprocessing
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
            
        # Preprocess
        processed_img = self.preprocess_image(image)
        
        # Perform OCR
        results = self.ocr.ocr(processed_img)
        
        return results, processed_img

def prepare_training_data(input_dir, output_dir):
    """
    Prepare training data for fine-tuning
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'images'))
        os.makedirs(os.path.join(output_dir, 'labels'))

    label_list = []
    for idx, img_name in enumerate(os.listdir(input_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path)
            
            # Save image with new name
            new_img_name = f'keyboard_{idx}.jpg'
            img.save(os.path.join(output_dir, 'images', new_img_name))
            
            # Create corresponding label file
            label_entry = {
                'filename': new_img_name,
                'text': 'your_ground_truth_text',  # Replace with actual text
                'bbox': [0, 0, img.width, img.height]  # Replace with actual bbox
            }
            label_list.append(label_entry)
    
    # Save labels
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(label_list, f, indent=2)

def create_custom_dict():
    """
    Create custom dictionary for keyboard characters
    """
    keyboard_chars = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '`', '-', '=', '[', ']', '\\', ';', "'", ',', '.', '/',
        'SHIFT', 'CTRL', 'ALT', 'SPACE', 'ENTER', 'TAB', 'CAPS', 'DEL'
    ]
    
    with open('your_custom_dict.txt', 'w') as f:
        for char in keyboard_chars:
            f.write(f"{char}\n")

def main():
    # Initialize KeyboardOCR
    keyboard_ocr = KeyboardOCR()
    
    # Example usage
    image_path = "your_keyboard_image.jpg"
    results, processed_img = keyboard_ocr.detect_keys(image_path)
    
    # Display results
    img = cv2.imread(image_path)
    for idx in range(len(results)):
        res = results[idx]
        for line in res:
            points = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Draw box
            points = np.array(points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points], True, (0, 255, 0), 2)
            
            # Add text
            cv2.putText(img, 
                       f"{text} ({confidence:.2f})",
                       tuple(points[0][0]),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 0, 0),
                       2)
    
    cv2.imshow('Processed Image', processed_img)
    cv2.imshow('Detection Results', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()