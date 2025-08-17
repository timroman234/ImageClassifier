# Import needed libraries

import tensorflow as tf      # TensorFlow is used for building and training neural networks

from tensorflow.keras import datasets, layers, models # Keras is a high-level API in TensorFlow

import matplotlib.pyplot as plt  # Matplotlib is used to display images and graphs

import os             # OS module is used to interact with the operating system



# Load and preprocess the CIFAR-10 dataset (collection of 60,000 32x32 color images in 10 classes)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# Normalize pixel values from [0,255] to [0,1] for easier training

train_images, test_images = train_images / 255.0, test_images / 255.0



# Define the names of the classes in CIFAR-10

class_names = [

  "airplane",

  "automobile",

  "bird",

  "cat",

  "deer",

  "dog",

  "frog",

  "horse",

  "ship",

  "truck",

]



# Set the path where the trained model will be saved and loaded

model_path = "cifar10_cnn_model.h5"



# Check if a saved model already exists

if os.path.exists(model_path):

  # Load the model from disk if available

  model = tf.keras.models.load_model(model_path)

  print("Model loaded from disk.")

else:

  # If not, build a new Convolutional Neural Network (CNN)

  model = models.Sequential([

    # First convolutional layer (extract features from images)

    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),

    # Pooling layer reduces spatial size and improves efficiency

    layers.MaxPooling2D((2, 2)),

    # Second convolutional layer

    layers.Conv2D(64, (3, 3), activation="relu"),

    layers.MaxPooling2D((2, 2)),

    # Third convolutional layer

    layers.Conv2D(64, (3, 3), activation="relu"),

    # Flatten layer converts 2D data to 1D array for dense layers

    layers.Flatten(),

    # Fully connected layer to learn more complex features

    layers.Dense(64, activation="relu"),

    # Output layer (10 units, one for each class)

    layers.Dense(10),

  ])



  # Compile the model (choose optimizer, loss function, and metric to track)

  model.compile(

    optimizer="adam",

    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

    metrics=["accuracy"],

  )



  # Train the model using training data (for 10 epochs), validate with test data

  model.fit(

    train_images,

    train_labels,

    epochs=10,

    validation_data=(test_images, test_labels),

  )



  # Save the trained model to disk, so you don’t have to retrain it every time

  model.save(model_path)

  print("Model saved to disk.")



# Define a function to classify a single image

def classify_image(image):

  img_array = tf.expand_dims(image, 0) # Add batch dimension to image (needed for prediction)

  predictions = model.predict(img_array) # Get model predictions

  predicted_class = tf.argmax(predictions[0]).numpy() # Find the index of class with highest probability

  return class_names[predicted_class] # Return the name of the predicted class



# Define a function to display an image with its predicted and true class labels

def show_image_with_prediction(image, true_label):

  predicted_label = classify_image(image)      # Predict the class of the given image

  plt.figure()

  plt.imshow(image)                 # Show the image

  plt.title(f"Predicted: {predicted_label}, True: {true_label}") # Show predicted and true labels

  plt.axis("off")                  # Hide axis for clarity

  plt.show()



# Classify the second test image

class_pred = classify_image(test_images[2])

# Display the third test image along with the predicted class

show_image_with_prediction(test_images, class_pred)

