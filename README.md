# Mongoltori Project

## Overview
A computer vision project utilizing YOLO and real-time keyboard detection methods. This project combines object detection and custom OCR techniques for keyboard analysis.

## Team
- Reshad
- Sammam
- Zamiul
- Fahim

## Prerequisites
- Python 3.x
- Anaconda
- YOLO (You Only Look Once)
- OpenCV

## Installation

### 1. Setting up the Environment
First, install Anaconda by following the guide in [Anaconda_guide](data/Setting_up_Anaconda_env_paddleOCR)

### 2. YOLO Installation
Install YOLO by following the instructions in [YOLO_guide](data/Yolo_guide.md)

### 3. Clone the Repository
```bash
git clone https://github.com/BRACU-Mongol-Tori/mongoltori_project.git
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
mongoltori_project/
├── data/
│   ├── Anaconda_guide.md
│   ├── YOLO_guide.md
│   └── Setting_up_Anaconda_env_paddleOCR.md
├── keyboard_key_detection/
│   ├── paddleOCR_iterative_method.py
│   ├── paddleOCR_non_iterative_method.py
│   ├── iterative_gridwise.py
│   └── test.py
├── main_src/
│   ├── keyboardRealtimeAngle.py
│   ├── main_detection_guidence_file.py
│   └── withDistance and angle.py
├── YOLO_training_and_models/
│   ├── pt_files/
│   ├── run_yolo.py
│   ├── train_yolo_model.py
│   └── train_yolo.yml
├── environment.yml
├── requirements.txt
└── README.md
```

## Key Features
1. Real-time keyboard detection
2. OCR implementation using PaddleOCR(R&D and Experimental)
   - Iterative method
   - Non-iterative method
   - Grid-wise detection
3. Distance and angle measurements
4. Custom YOLO training capabilities

## Usage

### 1. Keyboard Detection
#### This is out primary main detection and guidence file that we will use on our bot
```python
python main_src/main_detection_guidenece_file.py
```

### 2. Training Custom YOLO Model
```python
python YOLO_training_and_models/train_yolo_model.py
```

### 3. Running OCR Detection// OCR is very much into R&D now its NOT used in the bot currently.

```python
python keyboard_key_detection/paddleOCR_iterative_method.py
```

## Configuration
- Use `environment.yml` for custom environment settings that will help generalise the workspace for all users
- Adjust YOLO parameters in `train_yolo.yml`
- Configure OCR settings in respective detection files

## Development
1. Clone the repository
2. Create a short-lived feature branch from main (`git checkout -b feature-name`)
3. Keep changes small and focused
4. Push changes frequently (`git push origin feature-name`)
5. Create Pull Request to main branch
6. After review, merge and delete the feature branch

Note: We follow trunk-based development with main as our primary branch. Features should be small and merged frequently.




## Note
This project is part of BRACU Mongol Tori's initiatives. All rights are reserved by the BRACU Mongol Tori codebase team.