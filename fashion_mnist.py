# ==============================
# 1. Install Necessary Libraries
# ==============================

# Install tensorflow library
# Uncomment the following line if you need to install tensorflow

# !pip install tensorflow

# ==============================
# 2. Import Libraries
# ==============================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 3. Load the Fashion-MNIST Dataset
# ==============================

# Importing the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Dataset Classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Check the distribution of labels in the training data to see the dataset balance
unique, counts = np.unique(train_labels, return_counts=True)
label_distribution = dict(zip(unique, counts))

# Plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(class_names, counts)
plt.title("Label Distribution in Training Set")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Include Class Weight to ensure dataset is well balanced and distributed

from sklearn.utils import class_weight
# Calculate class weights to balance the data
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# ==============================
# 4. Data Preprocessing
# ==============================

# Normalizing the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to add a single channel (28x28x1)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Showing the images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# ==============================
# 5. Building the CNN Model with Six Layers
# ==============================
model = models.Sequential([
    # Layer 1: Convolutional Layer with 32 filters
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Layer 2: MaxPooling Layer
    layers.MaxPooling2D((2, 2)),

    # Layer 3: Convolutional Layer with 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Layer 4: Batch Normalization Layer
    layers.BatchNormalization(),

    # Layer 5: Flatten Layer
    layers.Flatten(),

    # Layer 6: Dense Layer with 128 units and Dropout
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # Output Layer: Dense Layer with 10 units for classification
    layers.Dense(10, activation='softmax')
])

# ==============================
# 6. Compiling the Model
# ==============================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==============================
# 7. Training the Model with class weights to deal with any possible dataset imbalance and Saving the training history for accuracy plots
# ==============================
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=64, 
                    validation_split=0.2,
                    class_weight=class_weights)

# ==============================
# 8. Plot Accuracy and Loss Curves
# ==============================
# Plotting training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Visualisation  Explained

# The validation accuracy improves alongside the training accuracy without diverging significantly, although the divergence from epoch 6 might indicate the beginning of overfitting, where the model starts to perform better on the training data than on unseen data. 
#  The loss plot similarly helps in diagnosing overfitting or underfitting by comparing the training and validation loss trends. The above loss plot corroborates the detail that overfitting might begin to occur as validation loss slightly diverges upwards.

# Checking the model accuracy and loss
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# ==============================
# 9. Make Predictions
# ==============================
# Make predictions on the first three test images
predictions = model.predict(test_images[:3])

# ==============================
# 10. Display Predictions with Confidence Plot
# ==============================

def plot_image_and_confidence(image, prediction, true_label):
    plt.figure(figsize=(8, 3))

    # Plot the image and prediction
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap=plt.cm.binary)
    plt.title(f"Predicted: {class_names[np.argmax(prediction)]}\nActual: {class_names[true_label]}")
    plt.axis('off')

    # Plot the confidence bar chart
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction, color="#777777")
    plt.xticks(range(10), class_names, rotation=90)
    plt.ylim([0, 1])
    plt.title('Confidence')
    plt.show()

# Display the first three test images with their predictions and confidence plots
for i in range(3):
    plot_image_and_confidence(test_images[i], predictions[i], test_labels[i])

# Visualisation Explained

# The confidence plots suggest the model has absolute certainty in predicting image selction with all 3 predicted images matching the actual image.