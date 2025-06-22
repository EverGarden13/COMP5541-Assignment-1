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

Or if you prefer JupyterLab:

```bash
jupyter lab
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

## Expected Runtime

- **Q2.ipynb**: 2-6 hours (depending on hardware and number of epochs)
- **Q3.ipynb**: 1-4 hours (transfer learning is generally faster)

*Note: Runtime significantly decreases with GPU acceleration*

## Outputs

Both notebooks will generate:
- Training/validation loss and accuracy plots
- Model performance comparisons
- Visualizations of training data
- Trained model weights (saved automatically)

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory Error**:
   - Reduce batch size in the notebooks
   - Use CPU instead of GPU by modifying device settings

2. **Package Import Errors**:
   - Ensure all required packages are installed
   - Check Python version compatibility

3. **Dataset Download Issues**:
   - Ensure stable internet connection
   - Clear `./data` folder and retry if download fails

4. **Slow Training**:
   - Reduce number of epochs for testing
   - Use GPU acceleration if available
   - Consider using smaller model variants

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

## Notes

- The CIFAR-10 dataset (~170MB) will be downloaded automatically on first run
- Models are trained for multiple epochs; you can interrupt and resume training
- Results may vary slightly due to random initialization
- For best results, run on a machine with GPU acceleration

## Support

If you encounter issues:
1. Check that all dependencies are correctly installed
2. Ensure sufficient disk space and memory
3. Verify Python/PyTorch versions are compatible
4. Try running with smaller batch sizes or fewer epochs for testing

## Assignment Context

These notebooks are part of COMP5541 Assignment 1, focusing on:
- Understanding CNN architectures
- Implementing transfer learning techniques
- Comparing different optimization strategies
- Analyzing model performance on image classification tasks 