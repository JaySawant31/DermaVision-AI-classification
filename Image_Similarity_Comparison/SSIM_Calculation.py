""" SSIM - Structural Similarity Index (Ranges from -1 to 1: higher the better!) """

#Import Necessary Libraries
import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score = ssim(img1, img2)
    print(f"SSIM: {score:.4f}")

compute_ssim("your_original_image_path_here.png", "your_generated_image_path here.png") #Replace with your respective original and generated image path
