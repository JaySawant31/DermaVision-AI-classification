#Import all the necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data(data_dir, transform):
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

# Check if a GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device) #Ensuring GPU usage

# Define data transforms (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ask the user for data directory paths
train_data_dir = "Your-Train-Data-Path here.pth"
test_data_dir = "Your-Test-Data-Path-here.pth"

# Load the data
train_loader = load_data(train_data_dir, transform)
test_loader = load_data(test_data_dir, transform)

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True) #Takes a ResNet-18 architecture pre-trained on the ImageNet Dataset
num_classes = len(train_loader.dataset.classes)  # Function to calculate the number of classes in your dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
num_epochs = 1000  # You can adjust this number as needed

train_accuracies = []  # To store training accuracies for each epoch
test_accuracies = []   # To store testing accuracies for each epoch
train_losses = []  # To store training losses for each epoch
test_losses = []  # To store testing losses for each epoch

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)

    # Testing
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    test_accuracy = 100 * test_correct / test_total
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')

# Calculate and print average training and testing accuracies
avg_train_accuracy = np.mean(train_accuracies)
avg_test_accuracy = np.mean(test_accuracies)

print(f'Average Training Accuracy: {avg_train_accuracy:.2f}%')
print(f'Average Testing Accuracy: {avg_test_accuracy:.2f}%')

"""Calculating more metrics for model performance evaluation"""
# Confusion Matrix and Classification Report
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

confusion = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(confusion)
report = classification_report(all_labels, all_preds, target_names=train_loader.dataset.classes)
print("Classification Report:")
print(report)

# Plot the accuracy and loss per epoch
plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(test_accuracies, label='Test Accuracy', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Testing Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()

