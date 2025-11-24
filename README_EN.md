# FalcoVision - Image Classification Project

This project implements an image classification system using PyTorch and ResNet18 architecture. It can classify images into different categories based on a trained model.

## Features

- Image classification using ResNet18 pre-trained model
- Transfer learning implementation
- Support for GPU acceleration
- Comprehensive training and testing scripts
- Data preprocessing and augmentation

## Project Structure

```
FalcoVision/
├── train.py          # Training script
├── test.py           # Testing/evaluation script
├── savedmodel.pth    # Saved trained model weights
├── readme.md         # This file
├── train/            # Training data directory (to be created by user)
│   ├── class1/       # Images for class 1
│   ├── class2/       # Images for class 2
│   └── ...           # Additional classes
└── test/             # Testing data directory (to be created by user)
    ├── class1/       # Test images for class 1
    ├── class2/       # Test images for class 2
    └── ...           # Additional classes
```

## Requirements

- Python 3.6+
- PyTorch
- TorchVision
- PIL (Pillow)

Install dependencies with:
```bash
pip install torch torchvision pillow
```

## Usage

### 1. Prepare Data

Organize your data in the following structure:
- Create a `train` directory with subdirectories for each class
- Create a `test` directory with the same subdirectory structure

Example:
```
train/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

### 2. Train the Model

Run the training script:
```bash
python train.py
```

The trained model will be saved as `savedmodel.pth`.

### 3. Test the Model

After training, evaluate the model on test data:
```bash
python test.py
```

## Configuration

### Training Parameters (in train.py)
- `num_epochs`: Number of training epochs (default: 25)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate for optimizer (default: 0.0005)

### Model Architecture
- Base model: ResNet18 (pre-trained on ImageNet)
- Custom classifier: Linear layer adapted to the number of classes

## Code Style

The code follows these conventions:
- PEP 8 style guide for Python
- English comments and documentation
- Descriptive variable and function names
- Comprehensive docstrings for functions
- Error handling for common issues

## License

This project is open-source and available under the MIT License.