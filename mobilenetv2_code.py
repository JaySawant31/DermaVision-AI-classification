#Import Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") #Ensuring the usage of GPU

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Define paths for training and testing data
train_folder = "your training dataset folder path"
test_folder = "your testing dataset folder path"

# Load the datasets
train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Take a pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
# Modify the output layer to match the number of classes in your dataset
num_classes = len(train_dataset.classes) #Stores the number of classes in a dataset (4 in this case!)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Move the model to GPU if available
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 500  # You can adjust this number according to your usage 
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / len(train_dataset)

    # Validation loop
    model.eval()
    test_loss = 0.0
    correct_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct_test / len(test_dataset)

    # Save metrics for plotting
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    # Print and log metrics
    print(f"Epoch {epoch + 1}/{num_epochs} => "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.4f}, "
          f"Test Accuracy: {test_accuracy:.4f}")

# Calculate average training and testing accuracy
avg_train_accuracy = sum(train_accuracies) / num_epochs
avg_test_accuracy = sum(test_accuracies) / num_epochs

print(f"\nAverage Training Accuracy: {avg_train_accuracy * 100:.2f}%")
print(f"Average Testing Accuracy: {avg_test_accuracy * 100:.2f}%")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Classification report
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=train_dataset.classes))

torch.save(model.state_dict(), 'MobileNetV2_model.pth') #Saving the updated weights of the model
