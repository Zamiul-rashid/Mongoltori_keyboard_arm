import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, det_db_thresh=0.5)

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(grayscale_image)
    blurred_image = cv2.GaussianBlur(inverted_image, (3, 3), 0)
    return blurred_image

def split_image_with_grid(image, rows, cols):
    segments = []
    h, w = image.shape[:2]
    segment_height = h // rows
    segment_width = w // cols
    for row in range(rows):
        for col in range(cols):
            y = row * segment_height
            x = col * segment_width
            segment = image[y:y + segment_height, x:x + segment_width]
            segments.append((segment, (x, y)))
    return segments, segment_height, segment_width

def perform_ocr_on_segments(segments):
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
                        if confidence > 0.3:
                            adjusted_points = [(int(p[0] + x), int(p[1] + y)) for p in points]
                            if text in segment_results:
                                if confidence > segment_results[text][2]:
                                    segment_results[text] = (adjusted_points, text, confidence)
                            else:
                                segment_results[text] = (adjusted_points, text, confidence)
    return segment_results

def perform_segmented_ocr(image_path, show_visualization=True):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_image = preprocess_image(img)
    all_results = {}

    segments_2x2, h2, w2 = split_image_with_grid(processed_image, 2, 2)
    results_2x2 = perform_ocr_on_segments(segments_2x2)
    all_results.update(results_2x2)

    segments_3x3, h3, w3 = split_image_with_grid(processed_image, 3, 3)
    results_3x3 = perform_ocr_on_segments(segments_3x3)
    for text, (points, txt, confidence) in results_3x3.items():
        if text in all_results:
            if confidence > all_results[text][2]:
                all_results[text] = (points, txt, confidence)
        else:
            all_results[text] = (points, txt, confidence)

    segments_6x4, h6, w4 = split_image_with_grid(processed_image, 6, 4)
    results_6x4 = perform_ocr_on_segments(segments_6x4)
    for text, (points, txt, confidence) in results_6x4.items():
        if text in all_results:
            if confidence > all_results[text][2]:
                all_results[text] = (points, txt, confidence)
        else:
            all_results[text] = (points, txt, confidence)

    # Draw OCR results
    for points, text, confidence in all_results.values():
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(img_rgb, [points_array], True, (0, 255, 0), 2)
        org = (int(points[0][0]), int(points[0][1] - 10))
        cv2.putText(img_rgb, f"{text}", org, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Draw grid lines on the image
    if show_visualization:
        for i in range(1, 2):  # 2x2 grid
            cv2.line(img_rgb, (i * w2, 0), (i * w2, img.shape[0]), (255, 0, 0), 2)
            cv2.line(img_rgb, (0, i * h2), (img.shape[1], i * h2), (255, 0, 0), 2)

        for i in range(1, 3):  # 3x3 grid
            cv2.line(img_rgb, (i * w3, 0), (i * w3, img.shape[0]), (0, 255, 0), 1)
            cv2.line(img_rgb, (0, i * h3), (img.shape[1], i * h3), (0, 255, 0), 1)

        for i in range(1, 6):  # 6x4 grid
            cv2.line(img_rgb, (i * w4, 0), (i * w4, img.shape[0]), (0, 0, 255), 1)
        for i in range(1, 4):
            cv2.line(img_rgb, (0, i * h6), (img.shape[1], i * h6), (0, 0, 255), 1)

        # Display the final annotated image with grid lines
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

    return list(all_results.values())

def print_results(results):
    print("\nDetected Text:")
    print("-" * 50)
    for _, text, confidence in results:
        print(f"Text: {text:<20} Confidence: {confidence:.2f}")
    print("-" * 50)

# Example usage
if __name__ == "__main__":
    image_path = r"D:\mongoltori_project\image copy.png"  # Replace with your image path
    results = perform_segmented_ocr(image_path)
    print_results(results)
