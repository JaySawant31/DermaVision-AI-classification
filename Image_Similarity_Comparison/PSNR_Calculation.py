""" PSNR - Peak Signal-to-Noise Ratio (Higher the better: typically >30 dB is considered good!) """

#Import Necessary Library
import cv2

def compute_psnr(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    score = cv2.PSNR(img1, img2)
    print(f"PSNR: {score:.2f} dB")

compute_psnr("your_original_image_path_here.png", "your_generated_image_path_here.png") # Replace with your respective original and AI-generated images
