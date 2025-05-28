## _Dermatological Disease Classification using Vision-based Deep Learning_


### Introduction

This repository contains all the codes for a project that aims to classifiy cutanoeus or skin diseases using pre-trained Deep-Learning models like ResNet-18, ResNet-50, VGG-16, MobileNetV2, and EfficientNetB0. The dataset is in the form of real-time taken skin image lesions. More information about the availabilty/procurement of the dataset is also provided. Data Augmentation techniques namely Basic Augmentation, Convolutional Autoencoder, and Wasserstein GANs to generate images have also been included in this repository since the initial dataset that is available is a very small one (50 images per class across 4 classes).

### What are Cutaneous Diseases?

Cutaneous Diseases or skin disorders are conditions that affect the skin, nails, and related tissues. Various factors are involved in causing these. 

### Problem Statement

Classification of skin diseases helps in early and accurate diagnosis, enabling timely and appropriate treatment. It also aids in standardizing care and improving clinical decision-making. Timely diagnosis becomes an important factor considering the fact that these diseases need to receive prompt medication to curb their spread. 

### Our Approach

We utilized real-time images of skin diseases and created a curated dataset to perform classification of cutaneous diseases based on four classes namely: _Herpes Simplex, Herpes Zoster, Molluscum Contagiosum, and Non-Viral_. The images were collected by Dr. Sushil Savant, MD Dermatologist, our collaborator in this project during his Masters Thesis at Katihar Medical College and Hospital, Bihar, India. Five neural network architectures whose names have been mentioned earlier were modified to be trained and tested on this curated dataset. A basic methodology pipeline adopted in this approach is depcited using a block diagram shown below:


![image](https://github.com/user-attachments/assets/de818def-b143-41f4-8e93-0b54c394a1e2)


_Refer to this paper published in ICPR 2024 for more information: https://doi.org/10.1007/978-3-031-78201-5_10_

### Dataset Availability

The dataset created specifically for this use is available on request at nitichikhale19@gmail.com or sushilsavant786@gmail.com. Make sure to include a short abstract or introduction to the research study/project you are going to perform using this data for us to evaluate the dataset requirement.

### Repository Contents

1. Different neural-network model fine-tuning codes saved as a Python file (.py) in [Deep_NN_Classifiers](https://github.com/JaySawant31/DermaVision-AI/tree/main/Deep_NN_Classifiers) are ready-to-use with the skin lesion dataset or with any dataset of the user's choice (make changes accordingly in some parts of the code if the dataset is different!)

2. Augmentation techniques' codes saved as a Python File (.py) in [Data_Augmentation_Methods](https://github.com/JaySawant31/DermaVision-AI/tree/main/Data_Augmentation_Methods) are available to generate images similar to the original ones

3. [Split_Data](https://github.com/JaySawant31/DermaVision-AI/tree/main/Split_Data) contains the data slitting code if in-case your dataset is not segregated between Training and Testing 

4. [Image_Similarity_Comparison](https://github.com/JaySawant31/DermaVision-AI/tree/main/Image_Similarity_Comparison) folder contains three different methods to perform comparison between original image and AI-generated image

5. All the pre-requisites (libraries download) for running the Python codes on your environment are stored in [Python_Pre-requisites](https://github.com/JaySawant31/DermaVision-AI/tree/main/Python_Pre-requisites)



