# 1. Install and Load Necessary Libraries

# Install keras if not already installed
if (!require(keras)) {
  install.packages("keras")
  library(keras)
}

# Load libraries
library(keras)
library(tensorflow)
library(tidyverse)

# 2. Load and Preprocess the Fashion MNIST Dataset

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Normalize images
train_images <- train_images / 255.0
test_images <- test_images / 255.0

# Reshape images to add a single channel (28x28x1)
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Define class names
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Calculate class weights
class_weights <- class_weight(train_labels)


# 3. Build the CNN Model

model <- keras_model_sequential() %>%
  # Layer 1: Convolutional Layer with 32 filters
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                input_shape = c(28, 28, 1)) %>%
  # Layer 2: MaxPooling Layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Layer 3: Convolutional Layer with 64 filters
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  # Layer 4: Batch Normalization Layer
  layer_batch_normalization() %>%
  # Layer 5: Flatten Layer
  layer_flatten() %>%
  # Layer 6: Dense Layer with 128 units and Dropout
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  # Output Layer: Dense Layer with 10 units for classification
  layer_dense(units = 10, activation = 'softmax')

# 4. Compile the Model

model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# 5. Train the Model

history <- model %>% fit(
  train_images, train_labels,
  epochs = 10, batch_size = 64,
  validation_split = 0.2,
  class_weight = class_weights
)


















