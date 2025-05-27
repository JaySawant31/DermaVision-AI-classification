#Import Libraries
import os
import cv2
import numpy as np

# Define the input and output directories
input_directory = "" # Replace with your input dataset folder path
output_directory = "" # Replace with the folder path you want to save your output data to

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the augmentation parameters (types of augmentations applied to the images)
rotation_angles = [0, 90, 180, 270]  # Rotation angles in degrees
horizontal_flip = True  # Horizontal flip
vertical_flip = True  # Vertical flip
skew_factors = [-0.5,-0.3,-0.2, 0.2, 0.3, 0.5]  # Skew factors
mirror=True

# Iterate through the classes (subfolders) in the input directory
for class_folder in os.listdir(input_directory):
    class_folder_path = os.path.join(input_directory, class_folder)

    # Create a subfolder in the output directory for the current class
    output_class_folder = os.path.join(output_directory, class_folder)
    if not os.path.exists(output_class_folder):
        os.makedirs(output_class_folder) #Make an output folder if it does not exist

    # Iterate through the images in the current class folder
    for filename in os.listdir(class_folder_path):
        image_path = os.path.join(class_folder_path, filename)

        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Apply rotation
        for angle in rotation_angles:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            output_image_path = os.path.join(output_class_folder, f"rotated_{angle}_{filename}")
            cv2.imwrite(output_image_path, rotated_image)

        # Apply horizontal flip
        if horizontal_flip:
            flipped_image = cv2.flip(image, 1)
            output_image_path = os.path.join(output_class_folder, f"horizontal_flip_{filename}")
            cv2.imwrite(output_image_path, flipped_image)

        # Apply vertical flip
        if vertical_flip:
            flipped_image = cv2.flip(image, 0)
            output_image_path = os.path.join(output_class_folder, f"vertical_flip_{filename}")
            cv2.imwrite(output_image_path, flipped_image)

        # Apply skew
        for skew_factor in skew_factors:
            pts1 = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]]])
            pts2 = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1] * skew_factor, image.shape[0]]])
            skew_matrix = cv2.getAffineTransform(pts1, pts2)
            skewed_image = cv2.warpAffine(image, skew_matrix, (image.shape[1], image.shape[0]))
            output_image_path = os.path.join(output_class_folder, f"skewed_{skew_factor}_{filename}")
            cv2.imwrite(output_image_path, skewed_image)

        # Apply mirror horizontally
        if mirror:
            mirrored_image = cv2.flip(image, 1)
            output_image_path = os.path.join(output_class_folder, f"mirrored_{filename}")
            cv2.imwrite(output_image_path, mirrored_image)

print("Image augmentation complete.") #Makes sure that the augmentation process is completed and all the augmented images are saved to the output folde
