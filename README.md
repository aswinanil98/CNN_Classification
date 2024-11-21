CNN-Based Shoe Classification
A deep learning project using Convolutional Neural Networks (CNN) and OpenCV (cv2) to classify images of shoes into two categories: Adidas and Nike.


Project Overview

This project demonstrates the use of CNNs for image classification. The model was trained to identify the brand of a shoe—either Adidas or Nike—based on its image. OpenCV was utilized for image preprocessing, and TensorFlow/Keras was used to build and train the CNN model.

Features

Binary classification of shoe images into Adidas or Nike.
Preprocessing of images using OpenCV for resizing, normalization, and augmentation.
Visualization of training metrics (accuracy and loss).
User-friendly interface to predict the brand of a shoe from an input image.

Dataset

The dataset contains labeled images of Adidas and Nike shoes.

Adidas: Images of shoes from Adidas.

Nike: Images of shoes from Nike.

Data Preparation

Images were resized to a consistent size (e.g., 128x128 pixels).
Augmentation techniques like flipping, rotation, and brightness adjustment were applied.
Dataset was split into training, validation, and testing sets for model evaluation.

Model Architecture

The CNN architecture used is composed of:

Convolutional Layers: For feature extraction.
Pooling Layers: For dimensionality reduction.
Fully Connected Layers: For classification.
Softmax Activation: For final binary classification.

Dependencies

Ensure the following Python libraries are installed:

TensorFlow/Keras
OpenCV (cv2)
NumPy
Matplotlib

How to Run the Project

Clone the Repository


Prepare the Dataset

Add the dataset to the data/ folder.

Train the Model

Training Accuracy: 98%
Test Accuracy: 98%

Key Learnings

Implementing CNNs for binary classification tasks.
Preprocessing and augmenting images using OpenCV.
Fine-tuning hyperparameters to optimize performance.

Future Enhancements

Expand the dataset to include more brands.
Improve accuracy by experimenting with advanced architectures (e.g., ResNet, MobileNet).
Deploy the model as a web app using Flask or Streamlit.

Acknowledgements

TensorFlow and Keras documentation for CNN implementation.
OpenCV for image preprocessing techniques.
