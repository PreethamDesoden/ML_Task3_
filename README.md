# CNN for Image Classification (TensorFlow/Keras)

## Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is built with TensorFlow/Keras and trained to classify images into 10 different categories. The final trained model achieves an accuracy of approximately 71% on the test dataset.

## Features

Built using TensorFlow/Keras.

Trained on CIFAR-10, a dataset with 60,000 color images (32x32 pixels) in 10 classes.

Implements multiple convolutional layers for feature extraction.

Uses MaxPooling to reduce dimensionality and ReLU activation for non-linearity.

Evaluates model performance using accuracy and loss metrics.

Supports visualization of predictions on test images.

Model can be saved and loaded for future use.

## Dataset: CIFAR-10

CIFAR-10 is a widely-used dataset in computer vision research. It consists of:

10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

50,000 training images and 10,000 test images.

Images are 32x32 pixels in RGB format.

## Installation

To run this project, install the required dependencies:

pip install tensorflow numpy matplotlib

## Model Architecture

The CNN model consists of:

Three Convolutional Layers with increasing filter sizes (32, 64, 128) and ReLU activation.

MaxPooling Layers after each convolutional layer to reduce spatial dimensions.

Flatten Layer to convert 2D feature maps into a 1D vector.

Fully Connected Dense Layer with 128 neurons.

Output Layer with 10 neurons (one for each class) using Softmax activation.

## Implementation Steps

Load and Preprocess Data

Normalize pixel values to range [0,1].

Split into training and test sets.

Build CNN Model

Stack convolutional, pooling, and dense layers.

Compile and Train Model

Use Adam optimizer and Sparse Categorical Crossentropy loss.

Train for 25 epochs.

Evaluate Model Performance

Check accuracy on test dataset.

Make Predictions

Test model with random images from the dataset.

Save the Model

Save the trained model for later use.

## Sample Prediction Output

After training, the model can predict an image category. Example output:

True: Cat | Predicted: Bird

## Results

Training Accuracy: ~75%

Test Accuracy: ~71%

## Saving & Loading Model

To save the trained model:

model.save("cnn_model.keras")

To load the model:

from tensorflow import keras
model = keras.models.load_model("cnn_model.keras")

## Future Improvements

Increase model depth with more layers.

Use data augmentation to enhance generalization.

Experiment with transfer learning using pre-trained models.

Tune hyperparameters like batch size, learning rate, and dropout.

## Conclusion

This project successfully implements a CNN-based image classifier using TensorFlow/Keras. The trained model achieves decent accuracy (71%) and demonstrates the power of deep learning for image recognition tasks.
