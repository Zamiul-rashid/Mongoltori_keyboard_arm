from paddleocr import PaddleOCR
import cv2
import matplotlib.pyplot as plt
import numpy as np

def perform_keyboard_ocr(image_path):
    """
    Perform OCR on an inverted contrast image of a keyboard using PaddleOCR
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        list: List of detected text and their coordinates
    """
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Invert the image contrast
    inverted_image = cv2.bitwise_not(img)
    # inverted_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise before edge detection
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

    # Display the inverted contrast image
    plt.title("Inverted Contrast")
    plt.imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Perform OCR on the inverted image
    results = ocr.ocr(inverted_image, cls=True)
    print(results)

    # Draw results on the original RGB image for visualization
    for idx in range(len(results)):
        res = results[idx]
        # print(res)
        for line in res:
            # Get coordinates
            points = line[0]
            text = line[1][0]  # The detected text
            confidence = line[1][1]  # Confidence score
            
            # Convert points to integers for drawing
            points = [(int(x), int(y)) for x, y in points]
            
            # Draw box
            cv2.polylines(img_rgb, [np.array(points)], True, (0, 255, 0), 2)
            
            # Add text above the box
            cv2.putText(img_rgb, 
                       f"{text} ({confidence:.2f})", 
                       (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       2)
    
    # Display the results
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    
    # Return the detected text and coordinates
    return results

def print_results(results):
    """
    Print the OCR results in a readable format
    
    Args:
        results (list): OCR results from PaddleOCR
    """
    print("\nDetected Text:")
    print("-" * 50)
    for idx in range(len(results)):
        res = results[idx]
        for line in res:
            text = line[1][0]
            confidence = line[1][1]
            print(f"Text: {text:<20} Confidence: {confidence:.2f}")
    print("-" * 50)

# Example usage
# path = r"/media/entropy/2E8E3A9F254EF703/mongoltori_project/image.png"
if __name__ == "__main__":
    image_path = r"/media/entropy/2E8E3A9F254EF703/mongoltori_project/image copy.png"  # Replace with your image path
    results = perform_keyboard_ocr(image_path)
    print_results(results)
