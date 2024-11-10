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
    return segments

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

def perform_segmented_ocr(frame):
    processed_image = preprocess_image(frame)
    all_results = {}
    segments_2x2 = split_image_with_grid(processed_image, 2, 2)
    results_2x2 = perform_ocr_on_segments(segments_2x2)
    all_results.update(results_2x2)

    segments_3x3 = split_image_with_grid(processed_image, 3, 3)
    results_3x3 = perform_ocr_on_segments(segments_3x3)
    for text, (points, txt, confidence) in results_3x3.items():
        if text in all_results:
            if confidence > all_results[text][2]:
                all_results[text] = (points, txt, confidence)
        else:
            all_results[text] = (points, txt, confidence)

    segments_6x4 = split_image_with_grid(processed_image, 6, 4)
    results_6x4 = perform_ocr_on_segments(segments_6x4)
    for text, (points, txt, confidence) in results_6x4.items():
        if text in all_results:
            if confidence > all_results[text][2]:
                all_results[text] = (points, txt, confidence)
        else:
            all_results[text] = (points, txt, confidence)

    for points, text, confidence in all_results.values():
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [points_array], True, (0, 255, 0), 2)
        org = (int(points[0][0]), int(points[0][1] - 10))
        cv2.putText(frame, f"{text}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def run_real_time_ocr():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = perform_segmented_ocr(frame)
        cv2.imshow("Real-Time OCR", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the real-time OCR
run_real_time_ocr()
