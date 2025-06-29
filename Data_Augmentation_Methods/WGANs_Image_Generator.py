#Import Required Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters to be selected for W GANS - select the best ones!
image_size = 64
channels = 3
latent_dim = 100
batch_size = 64
num_epochs = 100
lr = 0.00005
clip_value = 0.01
n_critic = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Loading the Data
def get_dataloader(data_folder):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*channels, [0.5]*channels)
    ])
    dataset = datasets.ImageFolder(root=data_folder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator Block
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.block(latent_dim, 512, 4, 1, 0),
            self.block(512, 256, 4, 2, 1),
            self.block(256, 128, 4, 2, 1),
            self.block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )

    def block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, z):
        return self.net(z)

# Discriminator Block
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            self.block(channels, 64, 4, 2, 1, bn=False),
            self.block(64, 128, 4, 2, 1),
            self.block(128, 256, 4, 2, 1),
            self.block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def block(self, in_c, out_c, k, s, p, bn=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p)]
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img).view(img.size(0), -1)

# Training Loop
def train_wgan(data_path, output_folder="generated_images"):
    dataloader = get_dataloader(data_path)
    G = Generator().to(device)
    D = Critic().to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=lr)
    optimizer_D = optim.RMSprop(D.parameters(), lr=lr)

    fixed_noise = torch.randn(64, latent_dim, 1, 1).to(device)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = imgs.to(device)

            # Training the Discriminator
            for _ in range(n_critic):
                z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
                fake_imgs = G(z).detach()
                loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # Weight Clipping
                for p in D.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Training the Generator
            z = torch.randn(imgs.size(0), latent_dim, 1, 1).to(device)
            gen_imgs = G(z)
            loss_G = -torch.mean(D(gen_imgs))

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        # Saving the generated images!
        if (epoch + 1) % 10 == 0:
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, f"{epoch+1:03d}.png")
            utils.save_image(gen_imgs.data[:25], save_path, nrow=5, normalize=True)

# Calling the main function to run the code
if __name__ == "__main__":
    data_folder = "your input folder path here"        # Replace with your dataset path
    save_folder = "your output folder path here"    # Replace with your desired output folder
    train_wgan(data_folder, output_folder=save_folder)
