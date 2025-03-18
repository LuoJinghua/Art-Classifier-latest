Art Style Classifier
Overview
The Art Style Classifier is a web application built using Streamlit that allows users to upload images of artwork and predict their art style using deep learning models. It supports multiple models, including EfficientNet, ResNet50, ResNet101, and ViT, and provides detailed results like the predicted art style, top-3 predictions, and a heatmap of class probabilities.

Features
Upload images in JPG, JPEG, or PNG formats.

Choose from multiple pre-trained models for prediction.

View the predicted art style, confidence score, top-3 predictions, and a heatmap of the top 10 class probabilities.

How to Use
Upload an image using the "Upload a picture" button.

Select a model from the sidebar dropdown (e.g., EfficientNet, ResNet50, ResNet101, or ViT).

View the prediction results, including the art style, confidence score, top-3 predictions, and heatmap.

Technologies
Streamlit: Web interface.

PyTorch: Deep learning models.

PIL (Pillow): Image preprocessing.

Matplotlib/Seaborn: Visualizations.

Timm: Model loading.

Models
EfficientNet-B2

ResNet50

ResNet101

ViT (Vision Transformer)

Dataset
Trained on the Artists Dataset, which includes artwork from various artists and styles.
