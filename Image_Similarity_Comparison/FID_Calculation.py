#Import Libraries
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

# The function to preprocess images for InceptionV3
def load_and_preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# User Input: Replace with paths to your images (Add the original and the generated image here)
image1_path = "your path to image1.jpg"  
image2_path = "your path to image2.jpg"  

# Loading and pre-processing the image 
img1 = load_and_preprocess(image1_path)
img2 = load_and_preprocess(image2_path)

# Calculate the FID between the two images given
fid = FrechetInceptionDistance(feature=2048, normalize=True) #2048 size feature vector extracted from InceptionV3's last layer
fid.update(img1, real=True)
fid.update(img2, real=False)
score = fid.compute()

print(f"Fréchet Inception Distance (FID) between the two images: {score.item():.4f}") #Printing the FID score on the o/p window
