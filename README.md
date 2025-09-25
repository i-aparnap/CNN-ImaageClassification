# CNN Image Classification – Intel Dataset

## Overview

This project builds a Convolutional Neural Network (CNN) to classify natural scene images into six categories: Buildings, Forest, Glacier, Mountain, Sea, Street. It covers data preprocessing, CNN model development, training, evaluation, and visualization.

## Dataset

Intel Image Classification Dataset (Kaggle)

25,000+ RGB images across 6 classes

Images resized to 150×150 and normalized for model input

## Preprocessing

Train/validation/test split

## Data augmentation 
rotation, flip, zoom, shift

Pixel values scaled to [0,1]

## CNN Architecture

3+ Convolutional layers with ReLU

Max pooling layers

Fully connected Dense layers with Dropout

Output layer with Softmax for 6 classes

Optimizer: Adam | Loss: Categorical Crossentropy

## Training & Evaluation

Early stopping and model checkpointing used

Metrics: Accuracy, Precision, Recall, F1-score

Confusion matrix and sample predictions visualized

## Results

High classification accuracy on test data

Confusion matrix shows class-wise performance

Predictions visualized for random test images

## How to Run

Open CNN Model for Image Classification.ipynb in Jupyter or Colab

Execute sequentially: preprocessing → model training → evaluation → visualization

## Dependencies

Python 3.8+, TensorFlow 2.x, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pillow,Keras

## Author

Aparna P
