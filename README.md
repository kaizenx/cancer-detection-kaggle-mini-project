# CNN Cancer Detection - Kaggle Mini-Project

This repository contains code for solving a binary image classification problem using a Convolutional Neural Network (CNN). The goal of this project is to detect metastatic cancer from histopathologic scans of lymph node sections.

## Problem Statement

We will use a CNN to identify whether cancer is present in images taken from digital pathology scans. The dataset for this project is provided by the [Histopathologic Cancer Detection competition on Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection/overview).

### Dataset

- **Source**: The dataset is available on Kaggle and includes approximately 220,025 images of size 96x96 pixels. Each image represents a histopathologic scan, and the goal is to classify them as either positive (cancerous) or negative (non-cancerous).
- **Format**: Images are provided in .tiff format, and the target labels are available in CSV files.

**Before running the code, download the dataset from the Kaggle competition and extract it into a directory named `data/`.**

## Project Structure

The project consists of a Jupyter notebook which performs the following tasks:

1. **Data Preprocessing**:
   - Resizes the images.
   - Converts images to tensors.
   - Normalizes images using specific mean and standard deviation values.
   
2. **Model Definition**:
   - Implements a CNN model tailored for binary classification.
   - Defines several convolutional and pooling layers to capture image features.
   
3. **Training**:
   - Trains the CNN model using the training dataset.
   - Utilizes techniques such as data augmentation to prevent overfitting.
   
4. **Evaluation**:
   - Monitors the model's performance on a validation set.
   - Detects overfitting based on the model's validation loss.
   
## Usage

### Prerequisites

To run the project, ensure you have the following libraries installed with CUDA enabled:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

