import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, device):
    """
    Train the image classification model.
    
    Args:
        model: The neural network model to train
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs (int): Number of training epochs
        dataloaders (dict): Dictionary containing train and validation data loaders
        dataset_sizes (dict): Dictionary containing sizes of train and validation datasets
        device: Device to run the training on (CPU or GPU)
        
    Returns:
        model: Trained model with best weights
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Set model to training mode
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over training data
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with gradient computation
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model if it has better accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete')
    print(f'Best validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    """Main function to train the image classification model."""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Define device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.0005
    
    # Data preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load data
    data_dir = './'
    
    # Check if train directory exists
    if not os.path.exists(os.path.join(data_dir, 'train')):
        raise FileNotFoundError("Train directory not found. Please ensure the 'train' folder exists with subfolders for each class.")
    
    num_classes = len(os.listdir('train'))
    print(f"Number of classes: {num_classes}")
    
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                       for x in ['train']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
        class_names = image_datasets['train'].classes
        print(f"Class names: {class_names}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define model
    print("Loading ResNet18 model...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train the model
    print("Starting training...")
    model = train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, device)
    
    # Save the trained model
    torch.save(model.state_dict(), 'savedmodel.pth')
    print("Model saved as 'savedmodel.pth'")


if __name__ == "__main__":
    main()