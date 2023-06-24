# ImageRecognition_Project
This project demonstrates image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to train a deep learning model to accurately classify images into their respective classes.

The project utilizes TensorFlow and Keras to build a convolutional neural network (CNN) model. It loads the CIFAR-10 dataset, preprocesses the images by normalizing pixel values, and splits the dataset into training and testing sets. The model is trained on the training set and evaluated on the testing set to measure its performance.

The trained model is then saved and later loaded for making predictions on new images. The project includes an example of predicting the class label of a user-uploaded image using the trained model.

Key Features:

Data preprocessing and normalization
Construction of a CNN model using TensorFlow and Keras
Training and evaluation of the model on the CIFAR-10 dataset
Saving and loading the trained model
Making predictions on user-uploaded images
Dependencies:

TensorFlow
Keras
NumPy
Matplotlib
OpenCV (for image loading and processing)
