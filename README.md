# Computer Vision Project from WorldQuant University
This project is part of a data science competition designed to help scientists track animals in a wildlife preserve. The objective of this project is to classify animals present in images taken by camera traps. The images are used as input for a machine learning model, and the task is to predict the species of the animal (if any) present in each image.

## Project Overview
The goal is to take images from camera traps and classify which animal, if any, is present in each image. The project involves using machine learning techniques, specifically deep learning with Convolutional Neural Networks (CNNs), to perform image classification.

### Key Learnings
Throughout this project, you'll gain hands-on experience with the following:

- Image Preprocessing: Learn how to read and prepare image files for use in machine learning tasks.
- PyTorch Framework: Get familiar with using PyTorch to manipulate tensors and build neural network models.
- Convolutional Neural Networks (CNNs): Build and train a CNN, a type of deep learning model particularly effective for image classification tasks.
- Model Evaluation and Predictions: Use the trained model to make predictions on new images and evaluate its performance.
- Competition Submission: Learn how to format and submit your model predictions for a data science competition.

### Project Goals
- Data Preparation: Load and preprocess the images from the camera traps.
- Model Building: Build a neural network model using PyTorch and train it on the labeled image data.
- Prediction: Use the trained model to predict the species of animals in unseen images.
- Submission: Format the predictions to create a submission for the competition.

### Technologies Used
- Python
- PyTorch: A deep learning framework for training neural networks.
- NumPy: For handling data manipulation.
- Pandas: For data handling and organization.
- Matplotlib: For data visualization.
- OpenCV / PIL: For image processing tasks.

### Steps Taken
- Data Exploration: Analyzed the dataset of images to understand its structure and prepare it for training.
- Model Building: Built a Convolutional Neural Network using PyTorch to classify the animal species in the images.
- Training: Trained the model with the dataset and evaluated its performance.
- Prediction: Used the trained model to classify animal species in new images.
- Submission: Created a formatted submission of predictions for the competition.

### How to Run
- Clone this repository to your local machine:
```sh
git clone https://github.com/temesgen5335/wildlife_conservation_with_computer_vision.git
```
- Install the required dependencies:
```sh
pip install -r requirements.txt
```
<!-- 
- Run the training and evaluation script:
```sh
python train.py
``` -->
License
This project is licensed under the MIT License - see the LICENSE file for details.