# MNIST Handwritten Digit Classification using CNN

A handwritten digit classification project using a Convolutional Neural Network (CNN) implemented with PyTorch on the MNIST dataset.

## Project Overview

The goal of this project is to classify handwritten digits from **0 to 9** using a CNN and achieve more than **99% test accuracy** on the standard MNIST test set.

The project follows this pipeline:

**MNIST → Preprocessing → CNN → Training → Evaluation → Error Analysis → Data Augmentation → Robustness Comparison**

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains:

* 60,000 training images
* 10,000 test images
* 10 classes: digits 0–9
* Image size: 28 × 28 pixels
* Grayscale images

The dataset is downloaded automatically using `torchvision.datasets.MNIST`.

## Preprocessing

The images are processed using:

* `ToTensor()` to convert pixel values to tensors and scale them to `[0, 1]`
* Normalization using the MNIST mean and standard deviation:

  * Mean: `0.1307`
  * Standard deviation: `0.3081`

Training images are loaded with a batch size of `128`.

## CNN Architecture

The model consists of two convolutional blocks followed by fully connected layers:

```text
Input: 1 × 28 × 28

Conv2D: 1 → 32, 3×3
ReLU
MaxPool 2×2

Conv2D: 32 → 64, 3×3
ReLU
MaxPool 2×2

Flatten

Linear: 3136 → 128
ReLU
Dropout: 0.5

Linear: 128 → 10
```

The final layer produces logits for the 10 digit classes.

## Training

The CNN is trained using:

* **Loss:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Initial learning rate:** `0.001`
* **Learning-rate scheduler:** StepLR
* **Step size:** 4 epochs
* **Gamma:** 0.5
* **Epochs:** 12
* **Random seed:** 42

The test set is used for evaluation after each epoch, but no checkpoint is selected based on test performance.

## Evaluation

The model is evaluated using:

* Test accuracy
* Classification report
* Confusion matrix
* Most common digit confusions
* High-confidence misclassifications

The project also visualizes the test examples where the model was most confident but predicted the wrong digit.

## Data Augmentation

A second CNN is trained using affine augmentation on the training data.

The augmentation includes:

* Rotation: ±15°
* Translation: up to 10%
* Scaling: 0.9×–1.1×

The augmented model is compared with the baseline model on:

1. Clean MNIST test images
2. A perturbed test set containing stronger affine transformations

This provides a comparison of both standard accuracy and robustness to image transformations.

## Project Structure

```text
.
├── mnist_cnn.ipynb
└── README.md
```

The notebook contains the complete preprocessing, model definition, training, evaluation, visualization, and augmentation experiments.

## Libraries Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## How to Run

Install the required libraries:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

Then open:

```text
mnist_cnn.ipynb
```

Run the notebook from top to bottom.

The MNIST dataset will be downloaded automatically during the first run.

## Results

The notebook evaluates both the baseline and augmented CNN models using clean and perturbed test sets.

The results include:

* Test accuracy
* Precision, recall and F1-score
* Confusion matrix
* Misclassified examples
* Baseline vs augmented model comparison

**Note:** Run the complete notebook before submission so that the final numerical results and visualizations are saved in the notebook.

## Key Learning Outcomes

This project demonstrates:

* Image preprocessing with PyTorch
* CNN architecture design
* Convolution and pooling
* ReLU activation
* Dropout regularization
* Cross-entropy loss
* Adam optimization
* Learning-rate scheduling
* Model evaluation
* Confusion-matrix analysis
* Error analysis
* Data augmentation
* Robustness testing

## Conclusion

This project demonstrates a complete deep-learning workflow for handwritten digit classification, from dataset preprocessing and CNN training to detailed evaluation and robustness analysis through data augmentation.
