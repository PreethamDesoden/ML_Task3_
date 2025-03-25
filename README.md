# CNN for Image Classification (TensorFlow/Keras)

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : PREETHAM B

*INTERN ID* : CT12WM72

*DOMAIN* : MACHINE LEARNING 

*DURATION* : 12 WEEKS

*MENTOR* : NEELA SANTOSH 

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

- Load and Preprocess Data

- Normalize pixel values to range [0,1].

- Split into training and test sets.

- Build CNN Model

- Stack convolutional, pooling, and dense layers.

- Compile and Train Model

- Use Adam optimizer and Sparse Categorical Crossentropy loss.

- Train for 25 epochs.

- Evaluate Model Performance

- Check accuracy on test dataset.

- Make Predictions

- Test model with random images from the dataset.

- Save the Model

- Save the trained model for later use.

## Sample Prediction Output

After training, the model can predict an image category. Example output:

True: Cat | Predicted: Bird

## Results

Training Accuracy: ~75%

Test Accuracy: ~71%

## Saving & Loading Model

To save the trained model:

model.save("cifar10_cnn_model.keras")

To load the model:

from tensorflow.keras.models import load_model

loaded_model = load_model("cifar10_cnn_model.keras")

## Future Improvements

Increase model depth with more layers.

Use data augmentation to enhance generalization.

Experiment with transfer learning using pre-trained models.

Tune hyperparameters like batch size, learning rate, and dropout.

## Conclusion

This project successfully implements a CNN-based image classifier using TensorFlow/Keras. The trained model achieves decent accuracy (71%) and demonstrates the power of deep learning for image recognition tasks.

## Output

![Image](https://github.com/user-attachments/assets/3ccd1210-fb99-4e8a-9852-e02c65c9c101)

![Image](https://github.com/user-attachments/assets/0e43a50c-fa20-4750-ae49-769bf01b6d04)

![Image](https://github.com/user-attachments/assets/30f3d938-ac46-4696-b6b1-063938be3a45)

![Image](https://github.com/user-attachments/assets/1710fec8-4d62-4176-aca6-9f7796725e89)

![Image](https://github.com/user-attachments/assets/fe208bad-60e7-4006-a875-477df4501e97)

![Image](https://github.com/user-attachments/assets/284ec1b7-9543-42f5-a63e-85fa14772da0)


