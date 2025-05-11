#Import Libraries
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import utils

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) #Ensuring GPU usage

# Custom dataset to load images from a single folder
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Convolutional Autoencoder architecture used for this approach
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Use Sigmoid for images in [0, 1] range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create a custom dataset and data loader
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
data_folder = "Path to your folder containing images"  # Replace with the path to your image folder
custom_dataset = CustomDataset(data_folder, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

# Initialize the autoencoder and move it to the device
autoencoder = Autoencoder().to(device)

# Define a loss function (MSE for image reconstruction)
criterion = nn.MSELoss() #Mean Squared Error Loss function used

# Define an optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)

# Lists to store loss values and epoch numbers
losses = []
epochs = []

# Training loop
num_epochs = 500 # Adjust this as per your usage 
for epoch in range(num_epochs):
    for data in data_loader:
        inputs = data.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    # Append loss and epoch number
    losses.append(loss.item())
    epochs.append(epoch + 1)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

torch.save(autoencoder.state_dict(), 'convolutional_autoencoder_model.pth') # Saving the CAE model path

# Generate images using the trained autoencoder
output_folder = 'path where you want to save the generated images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
num_images_to_generate = 50 # For user-defined image generation
with torch.no_grad():
    for i in range(num_images_to_generate):
        inputs = next(iter(data_loader)).to(device)
        generated_image = autoencoder(inputs)
        save_image(generated_image, os.path.join(output_folder, f'generated_image_{i}.png'))

# Display generated images
images = []
for i in range(num_images_to_generate):
    image_path = os.path.join(output_folder, f'generated_image_{i}.png')
    img = Image.open(image_path)
    images.append(transforms.ToTensor()(img))

#Display the generated images
grid = utils.make_grid(images, nrow=num_images_to_generate)
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()

#Epoch v/s Loss plotting
plt.figure(figsize=(8, 8))
plt.plot(epochs, losses, marker='o')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()
