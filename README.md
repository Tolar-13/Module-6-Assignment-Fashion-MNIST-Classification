# Fashion-MNIST Classification Using Convolutional Neural Networks (CNN)

## Overview

This project involves the implementation of Convolutional Neural Networks (CNN) using Python and R to classify images from the Fashion MNIST dataset. The dataset consists of 60,000 training images and 10,000 test images categorized into 10 classes, such as T-shirts, trousers, and shoes.

The CNN model built in both Python and R contains six layers and aims to achieve high accuracy in classifying the images.

## Python Implementation

### Steps:
1. **Load and Preprocess Data**: 
   - Data was loaded and normalized. 
   - Images were reshaped to include a channel dimension (28x28x1).

2. **Model Building**:
   - The CNN model includes two convolutional layers, a max-pooling layer, batch normalization, and two dense layers with dropout.

3. **Model Training**:
   - The model was trained over 10 epochs with a validation split of 20%.
   - Class weights were used to handle any imbalance in the dataset.

4. **Evaluation**:
   - The model achieved an accuracy of around 91% on the test dataset.

### Key Plots:
- **Accuracy and Loss Curves**: 
  - The training and validation accuracy showed an increasing trend, indicating good learning.
  - The validation loss slightly diverged from training loss in later epochs, indicating mild overfitting.

## R Implementation

### Steps:
1. **Install and Load Libraries**: 
   - Required libraries were installed and loaded, including `keras`, `tensorflow`, and `tidyverse`.

2. **Load and Preprocess Data**:
   - Data was loaded, normalized, and reshaped to the required dimensions.

3. **Model Building**:
   - A similar CNN model as in Python was built using the `keras` package in R.

4. **Model Training**:
   - The model was trained using the same parameters as the Python implementation.


## Insights and Suggestions

- **Overfitting**: The slight divergence of validation accuracy and loss suggests that the model might begin to overfit after a certain number of epochs. Regularization techniques such as early stopping, data augmentation, and more dropout layers could help reduce overfitting.
- **Improving Accuracy**: 
  - Further tuning of the model architecture, such as adding more convolutional layers or using transfer learning with pre-trained models, could enhance performance.
  - Additionally, increasing the number of training epochs with careful monitoring of validation performance might also improve results.

## Requirements

### Python:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter or any Python environment

### R:
- Keras
- TensorFlow
- Tidyverse
- RStudio or any R environment

Ensure all libraries are installed and available in your environment before running the code.

## How to run the code

### Python:
- Install and loaded the required libraries
- Run the python script `fashion_mnist.py` in the Python environment

### R:
- Install and loaded the required libraries
- Run the python script `fashion_mnist.r` in R studio


## Conclusion

Both Python and R implementations successfully built and trained CNN models to classify the Fashion MNIST dataset with over 90% accuracy. While the models performed well, there is room for improvement through the strategies discussed above.
