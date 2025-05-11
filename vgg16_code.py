#Import Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) #Ensuring usage of GPU for efficient computing

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = ImageFolder(root='your training dataset folder path here', transform=transform)
test_dataset = ImageFolder(root='your testing dataset folder path here', transform=transform)

# Define data loaders (Batch size can be adjusted accordingly)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Take a pre-trained VGG-16 model from torchvision.models
vgg16 = torchvision.models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False

# Modify classifier to match the number of classes
vgg16.classifier[-1] = nn.Linear(in_features=4096, out_features=len(train_dataset.classes))

# Move model to device (To use GPU)
vgg16.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.001)

# Training loop
num_epochs = 500 #Adjust according to your usage
train_losses = []
test_losses = []
train_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    vgg16.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy.append(correct_train / total_train)
    train_losses.append(running_loss / len(train_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {train_losses[-1]:.4f}, '
          f'Training Accuracy: {train_accuracy[-1]*100:.2f}%')

    # Testing loop
    vgg16.eval()
    running_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

        test_accuracy.append(correct_test / total_test)
        test_losses.append(running_loss / len(test_loader))

        print(f'Testing Loss: {test_losses[-1]:.4f}, '
              f'Testing Accuracy: {test_accuracy[-1]*100:.2f}%')

# Plotting the loss graph
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Testing loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

"""Calculate more metrics for model performance evaluation"""
print('Classification Report:')
print(classification_report(all_labels, all_predictions, target_names=train_dataset.classes))

print('Confusion Matrix:')
print(confusion_matrix(all_labels, all_predictions))

# Convert accuracies to percentages for better perception (Optional!)
train_accuracy_percent = [acc * 100 for acc in train_accuracy]
test_accuracy_percent = [acc * 100 for acc in test_accuracy]

# Calculate average accuracies
avg_train_accuracy = sum(train_accuracy) / len(train_accuracy)
avg_test_accuracy = sum(test_accuracy) / len(test_accuracy)

print(f'Average Training Accuracy: {avg_train_accuracy * 100:.2f}%')
print(f'Average Testing Accuracy: {avg_test_accuracy * 100:.2f}%')

torch.save(vgg16.state_dict(), 'vgg16_model.pth') #Save the model path with updated model weights
