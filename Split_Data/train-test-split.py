#Import Libraries
import os
import shutil
from sklearn.model_selection import train_test_split

"""Assume that the split is 80% for training the model and 20% for testing. You can adjust this accordingly!"""

def split_dataset(input_dir, output_dir, test_size=0.2): #This line ensures the train-test split is 80-20. You can also use train_size instead
    classes = os.listdir(input_dir)

    for class_name in classes:
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)

        for split, split_imgs in [('train', train_imgs), ('test', test_imgs)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)

            for img_name in split_imgs:
                src_path = os.path.join(class_path, img_name)
                dst_path = os.path.join(split_class_dir, img_name)
                shutil.copy2(src_path, dst_path)

    print("Dataset split completed and saved to:", output_dir) #Ensures the output folder is succesfully saved

input_dataset_dir = 'path/to/your/dataset' #Replace with input dataset folder path
output_dataset_dir = 'path/to/output/folder' #Replace with the folder path where you want to save your ouput dataset

split_dataset(input_dataset_dir, output_dataset_dir, test_size=0.2) #You can also use train_size=0.8
