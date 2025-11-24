import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def get_data_transforms():
    """
    Define data preprocessing transformations.
    
    Returns:
        Compose: Transformations to apply to images
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_class_names(data_dir):
    """
    Load and sort class names from directory structure.
    
    Args:
        data_dir (str): Path to the data directory
        
    Returns:
        tuple: (sorted class names list, number of classes)
    """
    class_names = sorted(os.listdir(data_dir))
    num_classes = len(class_names)
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}")
    return class_names, num_classes


def initialize_model(num_classes, model_path):
    """
    Initialize and load the trained model.
    
    Args:
        num_classes (int): Number of classification classes
        model_path (str): Path to the saved model weights
        
    Returns:
        model: Loaded PyTorch model
        device: Device the model is loaded on
    """
    # Load model architecture
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load saved weights
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model has been trained and saved.")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")
    
    # Set to evaluation mode
    model.eval()
    
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, device


def predict_image(image_path, model, data_transforms, device):
    """
    Predict the class of a single image.
    
    Args:
        image_path (str): Path to the image file
        model: Trained model for prediction
        data_transforms: Image transformations to apply
        device: Device to run prediction on
        
    Returns:
        int: Predicted class index
    """
    try:
        # Open and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = data_transforms(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        return preds[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def evaluate_model(test_dir, model, data_transforms, class_names, device):
    """
    Evaluate the model on test data.
    
    Args:
        test_dir (str): Path to test data directory
        model: Trained model for evaluation
        data_transforms: Image transformations to apply
        class_names (list): List of class names
        device: Device to run evaluation on
        
    Returns:
        tuple: (total samples, correct predictions, accuracy percentage)
    """
    total = 0
    correct = 0
    predicted_labels = []
    true_labels = []

    # Walk through test directory
    for root, _, files in os.walk(test_dir):
        for file in files:
            # Process only image files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                total += 1
                image_path = os.path.join(root, file)
                
                # Get true class from directory name
                true_class = os.path.split(root)[-1]
                
                # Make prediction
                predicted_class_idx = predict_image(image_path, model, data_transforms, device)
                
                if predicted_class_idx is not None:
                    predicted_class = class_names[predicted_class_idx]
                    
                    predicted_labels.append(predicted_class)
                    true_labels.append(true_class)

                    if predicted_class == true_class:
                        correct += 1
                    else:
                        print(f'Misclassified: {file} predicted as {predicted_class}, actual class is {true_class}')
                else:
                    print(f'Failed to process: {file}')

    # Calculate accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    return total, correct, accuracy


def main():
    """Main function to test the image classification model."""
    # Configuration
    test_dir = 'test'
    model_path = 'savedmodel.pth'
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory '{test_dir}' not found.")
    
    # Define data preprocessing
    data_transforms = get_data_transforms()
    
    # Load class names
    class_names, num_classes = load_class_names(test_dir)
    
    # Load model
    print("Loading model...")
    model, device = initialize_model(num_classes, model_path)
    
    # Evaluate model
    print("Evaluating model...")
    total, correct, accuracy = evaluate_model(test_dir, model, data_transforms, class_names, device)
    
    # Print results
    print(f'Total images: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()