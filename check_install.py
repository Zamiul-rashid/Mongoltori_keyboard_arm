import paddleocr; print(paddleocr.__version__)
from paddleocr import PaddleOCR; ocr = PaddleOCR(use_gpu=False); print('Installation successful')