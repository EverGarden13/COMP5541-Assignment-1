# COMP5541 Assignment 1 - Jupyter Notebooks

This repository contains two Jupyter notebooks for COMP5541 Assignment 1, implementing deep learning models for image classification on the CIFAR-10 dataset.

## Files Overview

### Q2.ipynb - Convolutional Neural Networks
Implements various CNN architectures for classifying CIFAR-10 images:
- **Part (a)**: Implementation and training of AlexNet, VGGNet, and ResNet architectures
- **Part (b)**: Training AlexNet with RMSProp and Adam optimizers
- **Part (c)**: Exploring methods to improve model performance

### Q3.ipynb - Transfer Learning
Explores transfer learning to improve model classification performance:
- Fine-tuning ImageNet pre-trained models (AlexNet, ResNet18, VGG16)
- Training with different amounts of data (10%, 20%, 50%)
- Comparing transfer learning vs. training from scratch
- Fine-tuning specific network layers

## Prerequisites

### Required Software
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (recommended for faster training)

### Required Python Packages
Install the following packages using pip:

```bash
pip install torch torchvision numpy matplotlib tqdm jupyter
```

Or install all dependencies at once:

```bash
pip install torch torchvision numpy matplotlib tqdm jupyter
```

### For Conda users:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy matplotlib tqdm jupyter
```

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU with 4+ cores
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, CUDA 11.8+
- **Storage**: At least 2GB free space for CIFAR-10 dataset and model weights

## How to Run

### 1. Clone/Download the Repository
Ensure you have the following files in your working directory:
- `Q2.ipynb`
- `Q3.ipynb`

### 2. Install Dependencies
Run the installation commands mentioned in the Prerequisites section.

### 3. Start Jupyter Notebook
Open a terminal/command prompt in the directory containing the notebooks and run:

```bash
jupyter notebook
```
### 4. Open and Run the Notebooks

#### For Q2.ipynb (CNN Architectures):
1. Open `Q2.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook will automatically:
   - Download CIFAR-10 dataset (first run only)
   - Load and preprocess the data
   - Implement AlexNet, VGGNet, and ResNet architectures
   - Train models with different optimizers
   - Display training progress and results

#### For Q3.ipynb (Transfer Learning):
1. Open `Q3.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook will automatically:
   - Download CIFAR-10 dataset (if not already present)
   - Load pre-trained models from PyTorch Model Zoo
   - Fine-tune models with different data portions
   - Compare transfer learning vs. training from scratch
   - Display comparative results


## Outputs

Both notebooks will generate:
- Training/validation loss and accuracy plots
- Model performance comparisons
- Visualizations of training data
- Trained model weights (saved automatically)


### GPU Setup
To use GPU acceleration:
1. Install CUDA-compatible PyTorch version
2. Verify GPU is detected: `torch.cuda.is_available()` should return `True`
3. The notebooks automatically detect and use GPU if available

## File Structure After Running

```
COMP5541-Assignment-1/
├── Q2.ipynb
├── Q3.ipynb
├── data/
│   └── cifar-10-python.tar.gz
│   └── cifar-10-batches-py/
└── (generated model files and plots)
```

