
# CIFAR-10 Image Classifier

A beginner-friendly project for image classification with TensorFlow and Keras, using the **CIFAR-10** dataset. This repository trains a simple CNN (Convolutional Neural Network) to recognize 10 different categories of small color images.

***

## Features

- Loads and preprocesses the CIFAR-10 dataset.
- Builds a CNN or loads a pre-trained model from disk.
- Classifies images into 10 categories.
- Displays sample predictions with image visualization.

***

## Requirements

- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/) (2.x)
- [Matplotlib](https://matplotlib.org/)
- (Optional) GPU support for faster training.

***

## Installation

```bash
git clone https://github.com/your-username/cifar10-image-classifier.git
cd cifar10-image-classifier

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow matplotlib
```


***

## Usage

1. **Train or Load the Model**
    - On first run, the script will train the CNN and save it as `cifar10_cnn_model.h5`.
    - On subsequent runs, it will load the model from disk for faster startup.
2. **Classify and Visualize Images**
    - The script predicts the class of a sample test image and displays the image with its predicted label.
```bash
python cifar10_classifier.py
```


***

## Script Overview

- **Data Loading:** Uses Keras to automatically download CIFAR-10 images and labels.
- **Preprocessing:** Scales pixel values to the  range.
- **Model Architecture:** Three convolutional layers with max-pooling, followed by dense layers and a 10-unit output.
- **Saving \& Loading:** Model is saved after training, loaded if available.
- **Prediction:** Includes a function for predicting the class of an image and visualizing the result.

***

## Classes

| Index | Class |
| :-- | :-- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |


***

## Example Output

On running the script, you’ll see something like:

> Model saved to disk.
> Displays an image with: "Predicted: cat, True: cat"

***

## License

This project is licensed under the MIT License.

***

## Contact

Created by [Your Name].
Feel free to open issues or submit pull requests!

***

**Happy Learning!**

