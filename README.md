# Pneumonia Detection using CNN

## Overview

This project detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). It provides an end-to-end pipeline including model training, evaluation, and deployment through a web interface.

## Features

* Image classification (Normal vs Pneumonia)
* Custom CNN architecture
* Model training and evaluation
* Flask-based web application
* Automated dataset download setup

## Setup Instructions

### 1. Install dependencies

pip install -r requirements.txt

### 2. Download dataset

python downloads/setup.py

### 3. Train the model

python train.py

### 4. Run the application

python app.py

## Dataset

Dataset used:
Chest X-Ray Pneumonia Dataset (Kaggle)

The dataset is not included in this repository due to size limitations.

## Project Structure

* train.py → Model training
* app.py → Web application
* predict.py → Image prediction
* downloads/setup.py → Dataset download script

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Flask

## Output

The model predicts:

* NORMAL
* PNEUMONIA

## Note

This project is for educational purposes and demonstrates the application of deep learning in medical image classification.
