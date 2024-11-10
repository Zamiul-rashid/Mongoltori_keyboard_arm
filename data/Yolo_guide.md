# YOLOv8 Installation Guide

This guide will help you install CUDA, YOLOv8, and the Ultralytics package on both Ubuntu and Windows.

## Prerequisites

- Ubuntu 18.04 or later
- Windows 10 or later
- Python 3.6 or later
- NVIDIA GPU with CUDA support

## Step 1: Install CUDA

### On Ubuntu

1. **Update and upgrade your system:**
    ```bash
    sudo apt update
    sudo apt upgrade
    ```

2. **Install CUDA Toolkit:**
    - Download the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
    - Follow the installation instructions provided on the website.

3. **Verify CUDA installation:**
    ```bash
    nvcc --version
    ```
    **If this isn't working check if you added the path to you env_variable/.bashrc file**
 
### On Windows

1. **Download and install CUDA Toolkit:**
    - Download the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
    - Follow the installation instructions provided on the website.

2. **Verify CUDA installation:**
    - Open Command Prompt and run:
    ```cmd
    nvcc --version
    ```
    **If this isn't working check if you added the path to you env_variable/.bashrc file**
# You don't to install cuDNN if you install Pytorch of the specific version of cuda you installed
## Step 2: Install PyTorch with GPU Support

### On Ubuntu

1. **Install PyTorch with CUDA:**
    - Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.
    - For example, to install PyTorch with CUDA 11.3, run:
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```

### On Windows

1. **Install PyTorch with CUDA:**
    - Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.
    - For example, to install PyTorch with CUDA 11.3, open Command Prompt and run:
    ```cmd
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
## Step 2: Install cuDNN


### On Ubuntu

1. **Download cuDNN:**
    - Go to the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn) and download the appropriate version for your CUDA installation.

2. **Install cuDNN:**
    ```bash
    tar -xzvf cudnn-*-linux-x64-v*.tgz
    sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```

### On Windows

1. **Download cuDNN:**
    - Go to the [NVIDIA cuDNN website](https://developer.nvidia.com/cudnn) and download the appropriate version for your CUDA installation.

2. **Install cuDNN:**
    - Extract the files and copy the contents to the CUDA installation directory, typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`.

## Step 3: Install Python and pip

### On Ubuntu

1. **Install Python:**
    ```bash
    sudo apt install python3
    ```

2. **Install pip:**
    ```bash
    sudo apt install python3-pip
    ```

### On Windows

1. **Install Python:**
    - Download the Python installer from the [official Python website](https://www.python.org/downloads/windows/).
    - Run the installer and follow the instructions. Make sure to check the box that says "Add Python to PATH".

2. **Install pip:**
    - Open Command Prompt and run:
    ```cmd
    python -m ensurepip --upgrade
    ```

## Step 4: Install YOLOv8 and Ultralytics

### On Ubuntu and Windows

1. **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv yolov8_env
    source yolov8_env/bin/activate
    ```

2. **Install Ultralytics package:**
    ```bash
    pip install ultralytics
    ```

3. **Verify YOLOv8 installation:**
    ```bash
    yolo
    ```

## Step 5: Test YOLOv8

1. **Run a test script:**
    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO('yolov8n.pt')

    # Perform inference
    results = model('path/to/your/image.jpg')

    # Print results
    results.show()
    ```

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/yolov8)

## Troubleshooting

- Ensure your GPU drivers are up to date.
- Verify that your CUDA and cuDNN versions are compatible.

For further assistance, refer to the official documentation or community forums.

