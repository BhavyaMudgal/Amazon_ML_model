# Smart Product Pricing Challenge

## Overview

This project was developed for the ML Challenge 2025 - Smart Product Pricing Challenge, where the objective was to predict product prices using product descriptions, packaging information, and product images.

The solution follows a multimodal machine learning approach, combining Natural Language Processing (NLP), Computer Vision, and Gradient Boosting techniques to learn complex relationships between product characteristics and their market prices.

---

## Problem Statement

Accurate product pricing is critical for e-commerce platforms. Product prices are influenced by various factors including:

- Brand information
- Product specifications
- Product quantity and packaging
- Visual appearance
- Product category and description

The goal of this challenge is to build a machine learning model capable of predicting product prices using only the provided dataset.

---

## Dataset

### Training Data
- 75,000 labeled products

### Test Data
- 75,000 unlabeled products

### Features

| Feature | Description |
|----------|------------|
| sample_id | Unique product identifier |
| catalog_content | Product title, description, and Item Pack Quantity (IPQ) |
| image_link | Product image URL |
| price | Target variable (training only) |

---

# Solution Architecture

text Catalog Content ──┐                   │                   ├── TF-IDF Features                   │                   ├── MiniLM Embeddings                   │                   └── Item Pack Quantity                               │                               ▼                     Feature Consolidation                               ▲                               │ Product Images ──► EfficientNet-B0 Features                               │                               ▼                      LightGBM Regressor                               │                               ▼                      Price Prediction 

---

## Feature Engineering

### 1. Text Features

#### TF-IDF Vectorization

The product descriptions were transformed using:

- Unigrams
- Bigrams
- Maximum 10,000 features

This allowed the model to capture important pricing-related keywords and phrases.

---

### 2. Semantic Text Embeddings

To capture contextual meaning beyond keyword frequency, product descriptions were encoded using:

Sentence Transformer
- Model: paraphrase-MiniLM-L6-v2

These dense embeddings provide semantic representations of product information.

---

### 3. Custom Numerical Features

A custom feature extraction pipeline was implemented to extract:

- Item Pack Quantity (IPQ)

The extracted quantity information was standardized and included in the final feature set.

---

## Computer Vision Pipeline

### EfficientNet-B0 Feature Extraction

Product images were processed using a pretrained:

- EfficientNet-B0

The final classification layer was removed, and image embeddings were extracted from the backbone network.

Image preprocessing included:

- Resize to 224×224
- Normalization using ImageNet statistics
- Batch inference on GPU

These embeddings capture visual cues related to:

- Product category
- Packaging
- Brand appearance
- Product quality indicators

---

## Model Training

### Target Transformation

Since product prices exhibit a highly skewed distribution, the target variable was transformed using:

python log_price = log1p(price) 

This improved model stability and prediction performance.

---

### Regression Model

The final model uses:

LightGBM Regressor

Reasons for selection:

- Handles high-dimensional sparse features efficiently
- Works well with heterogeneous feature types
- Fast training and inference
- Strong performance on tabular data

---

## Hyperparameter Optimization

Hyperparameters were optimized using:

Optuna

Optimized parameters included:

- Learning Rate
- Number of Leaves
- Maximum Depth
- Feature Fraction
- Bagging Fraction
- L1 Regularization
- L2 Regularization

This resulted in improved cross-validation performance.

---

## Validation Strategy

To ensure robust evaluation:

### 5-Fold Cross Validation

- KFold Cross Validation
- Shuffle Enabled
- Fixed Random Seed

This reduces overfitting risk and provides reliable performance estimates.

---

## Evaluation Metric

The challenge metric is:

### SMAPE (Symmetric Mean Absolute Percentage Error)

[
SMAPE = \frac{1}{n}\sum \frac{|y_{pred}-y_{true}|}{(|y_{true}|+|y_{pred}|)/2}
]

Lower values indicate better performance.

---

## Technologies Used

### Machine Learning
- LightGBM
- Scikit-Learn
- Optuna

### Natural Language Processing
- Sentence Transformers
- TF-IDF

### Computer Vision
- PyTorch
- TorchVision
- EfficientNet-B0

### Data Processing
- Pandas
- NumPy

### Experimentation
- Jupyter Notebook
- Google Colab

---

## Key Highlights

- Multimodal feature engineering combining text, image, and numerical information
- Semantic representation using transformer-based embeddings
- Deep image feature extraction using EfficientNet-B0
- Automated hyperparameter optimization using Optuna
- Robust evaluation using 5-Fold Cross Validation
- Price distribution normalization through log transformation

---

## Future Improvements

Potential enhancements include:

- End-to-end multimodal neural networks
- Fine-tuning EfficientNet on product images
- Incorporating vision-language models such as CLIP
- Ensemble learning with CatBoost and XGBoost
- Advanced entity extraction from product descriptions

---

## Author

Developed as part of ML Challenge 2025 to explore multimodal machine learning techniques for large-scale product price prediction in e-commerce environments.
