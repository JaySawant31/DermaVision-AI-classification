#Import libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) #Ensuring the usage of a GPU for efficient computing

# Define transforms (A standard 224 X 224 image transform size chosen)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
train_dataset = torchvision.datasets.ImageFolder(root='your training dataset path here', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4) #You can adjust the hyperparameters

test_dataset = torchvision.datasets.ImageFolder(root='your testing dataset path here', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Import a pre-trained ResNet-50 model from torchvision.models
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss() #Used the CE loss for this application, you may use another one if necessary
optimizer = optim.Adam(model.parameters(), lr=0.0001) #Change this to select the correct learning rate

# Training function
def train_model(model, criterion, optimizer, num_epochs=500):
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(train_loader))
        train_accuracy.append(correct_train / total_train * 100)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        test_accuracy.append(correct_test / total_test * 100)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, "
              f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracy[-1]:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_model_WGans_500.pth')

    # Plotting loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Average accuracy
    avg_train_accuracy = np.mean(train_accuracy)
    avg_test_accuracy = np.mean(test_accuracy)
    print(f"Average Train Accuracy: {avg_train_accuracy:.2f}%")
    print(f"Average Test Accuracy: {avg_test_accuracy:.2f}%")

    return train_accuracy, test_accuracy, avg_train_accuracy, avg_test_accuracy

# Train the model
train_accuracy, test_accuracy, avg_train_accuracy, avg_test_accuracy = train_model(model, criterion, optimizer, num_epochs=500)

"""Calculate more metrics for model performance evaluation"""
# Calculate confusion matrix and classification report
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

