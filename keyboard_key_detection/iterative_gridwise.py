from paddleocr import PaddleOCR
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, det_db_thresh=0.5)

def preprocess_image(image):
    # Grayscale, invert colors, and apply slight Gaussian blur
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(grayscale_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (3, 3), 0)
    return blurred_image

def split_image_with_grid(image, rows, cols):
    """Split the image into segments based on the specified grid dimensions (rows x cols)."""
    segments = []
    h, w = image.shape[:2]
    segment_height = h // rows
    segment_width = w // cols

    # Create segments for the given grid
    for row in range(rows):
        for col in range(cols):
            y = row * segment_height
            x = col * segment_width
            segment = image[y:y + segment_height, x:x + segment_width]
            segments.append((segment, (x, y)))
    return segments

def perform_ocr_on_segments(segments):
    """Perform OCR on each segment and return results with position adjusted."""
    segment_results = {}

    for segment, (x, y) in segments:
        ocr_results = ocr.ocr(segment, cls=True)

        if ocr_results:
            for res in ocr_results:
                if res:
                    for line in res:
                        points = line[0]
                        text = line[1][0]
                        confidence = line[1][1]

                        # Only consider results with confidence > 0.3 for reliability
                        if confidence > 0.3:
                            adjusted_points = [(int(p[0] + x), int(p[1] + y)) for p in points]
                            if text in segment_results:
                                if confidence > segment_results[text][2]:  # Keep only highest confidence
                                    segment_results[text] = (adjusted_points, text, confidence)
                            else:
                                segment_results[text] = (adjusted_points, text, confidence)
    return segment_results

def perform_segmented_ocr(image_path, max_rows=6, show_visualization=True):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Dictionary to hold aggregated results
    aggregated_results = defaultdict(lambda: {"points": None, "total_confidence": 0, "count": 0})

    # Loop through grid configurations from 1x1 up to max_rows x (max_rows-2)
    for rows in range(1, max_rows + 1):
        for cols in range(1, rows + 1):
            segments = split_image_with_grid(processed_image, rows, cols)
            results = perform_ocr_on_segments(segments)

            # Aggregate results across different grids
            for text, (points, txt, confidence) in results.items():
                if aggregated_results[text]["points"] is None:
                    aggregated_results[text]["points"] = points
                aggregated_results[text]["total_confidence"] += confidence
                aggregated_results[text]["count"] += 1

    # Calculate average confidence for each character
    final_results = {}
    for text, data in aggregated_results.items():
        avg_confidence = data["total_confidence"] / data["count"]
        final_results[text] = (data["points"], text, avg_confidence)

    # Draw the final results on the image with aggregated confidence
    for points, text, confidence in final_results.values():
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(img_rgb, [points_array], True, (0, 255, 0), 2)
        org = (int(points[0][0]), int(points[0][1] - 10))
        cv2.putText(img_rgb, f"{text}: {confidence:.2f}",
                    org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display the final annotated image if visualization is enabled
    if show_visualization:
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    return list(final_results.values())

def print_results(results):
    print("\nDetected Text:")
    print("-" * 50)
    for points, text, confidence in results:
        if isinstance(points, list) and len(points) == 4:
            coordinates = [(p[0], p[1]) for p in points]
            print(f"Text: {text:<10} Confidence: {confidence:.2f} Coordinates: {coordinates}")
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    image_path = r"D:\mongoltori_project\image2.png"  # Replace with your image path
    results = perform_segmented_ocr(image_path)
    print_results(results)
