# 🚦 Stop Sign Classifier with CNN

A deep learning project for binary image classification: detecting whether an image contains a **STOP sign** or not using a Convolutional Neural Network (CNN) based on ResNet18.
---
Please open the notebook **traffic-signal-classifier.ipynb** as there is a step-by-step explanation of the project.

## 📌 Overview

This project implements a complete pipeline for training and evaluating a CNN model:

- Dataset preparation and splitting
- Model training and validation
- Hyperparameter tuning
- Performance visualization (loss & accuracy)
- Model saving and loading
- Testing

The model classifies images into:

- 🛑 Stop sign  
- 🚫 Non-stop sign  

---

## 🧠 Model Architecture

- Backbone: ResNet18
- Custom fully connected layer
- Loss function: CrossEntropyLoss
- Optimizer: SGD with momentum
- Scheduler: Cyclic Learning Rate

---

## 📂 Project Structure

```bash
stop-sign-classifier/
│
├── data/
│   ├── classes/
│       ├── not_stop/
│       └── stop/
│   ├── datasets/
│       ├── train/
│       └── val/
│   ├── raw/
│       ├── raw_images/
│       ├── test_images/
│       └── _annotations.json
│
├── src/
│   ├── __init__.py
│   ├── cnn_manager.py
│   └── dataset.py
│
├── outputs/
│   ├── models/
│   └── plots/
│
├── requirements.txt
└── README.md
```

## 📚 Acknowledgements

This project was developed using concepts and guidance from the  course **Introduction to Computer Vision and Image Processing by IBM (Coursera)**, which provided foundational knowledge on convolutional neural networks, image preprocessing, and model training workflows.

Some parts of the implementation and experimental structure are inspired by the course material.
