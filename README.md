
# Vehicle Quality Classification Using Streamlit and Naive Bayes

## Introduction
Vehicle quality assessment can be a subjective task—yet it’s crucial for manufacturers, dealers, and buyers who rely on consistent, data-driven insights. In this project, we build a vehicle classification application using Streamlit for an interactive interface and a Categorical Naive Bayes model for classifying the quality or acceptability of a car based on various categorical features (e.g., price, maintenance cost, trunk size, etc.).

By the end of this tutorial, you will see how to load data, preprocess it, train a classification model, and create a simple web application that allows users to input vehicle features and instantly see a prediction of the car’s quality rating.

## The Dataset
Here, we read a CSV file containing categorical data about cars. The relevant columns are:
- buying: Price of the car (e.g., “low,” “high,” “vhigh”).
- maint: Maintenance cost.
- doors: Number of doors (2, 3, 4, 5+).
- persons: Seating capacity (2, 4, 5+).
- lug_boot: Trunk size (small, med, big).
- safety: Safety rating (low, med, high).
- class: The target variable representing the car’s acceptability (unacc, acc, good, vgood).

## Overview of the Algorithm
Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It calculates the probability of a class given a set of features. The "naive" assumption lies in treating all features as independent of each other, given the class. Despite this simplification, Naive Bayes often exhibits surprisingly good performance, particularly with categorical data.

## When to Use Categorical Naive Bayes
Categorical Naive Bayes is particularly well-suited for scenarios where:
- Data primarily consists of categorical features: This aligns with its core strength of handling discrete variables effectively.
- Speed and efficiency are crucial: Training is fast, making it ideal for real-time applications or situations where rapid model iteration is necessary.
- Interpretability is a priority: The probabilistic nature of the model provides insights into the factors contributing to the classification.
- Dataset size is moderate: It performs well with smaller to medium-sized datasets, both computationally and memory-wise.

## When to Avoid Categorical Naive Bayes
- Strong Feature Dependencies: When features are highly correlated, the independence assumption can significantly degrade performance.
- Continuous Features: While adaptations exist, Naive Bayes is generally less suitable for continuous features compared to algorithms designed for this data type (e.g., Gaussian Naive Bayes).
- Imbalanced Datasets: In cases of severe class imbalance, the model might be biased towards the majority class.
- High Accuracy is Paramount: If the primary goal is achieving the highest possible accuracy, more sophisticated models (e.g., Support Vector Machines, Random Forests) might outperform Naive Bayes.

## Key Benefits
- Efficiency: Rapid training enables quick experimentation and model iteration.
- Interpretability: Provides probabilistic outputs that are easily understandable by both technical and non-technical audiences.
- Ease of Implementation: Relatively simple to implement and deploy in production environments.

## Examples of use
- Spam Filtering: Classifying emails as spam or not spam based on the presence of certain words, phrases, and sender addresses.
- Sentiment Analysis: Determining the sentiment (positive, negative, neutral) expressed in customer reviews or social media posts.
- Disease Diagnosis: Assisting in medical diagnosis based on patient symptoms (categorical).

## In Summary
Categorical Naive Bayes offers a valuable balance of simplicity, efficiency, and interpretability. It's a strong contender for classification tasks involving categorical data, especially when speed and interpretability are paramount. However, its performance can be impacted by strong feature dependencies and imbalanced datasets.


# Model Training and Evaluation

Below is a concise overview of the training pipeline:

## Load and Preprocess

- **Loading**: Read the CSV dataset into a pandas DataFrame.
- **Categorical Encoding**: Convert features to category datatype.
- **Ordinal Encoding**: Use OrdinalEncoder to transform the categorical features into numeric codes (one integer per category).

## Train-Test Split

- **Splitting**: Use `train_test_split` to hold out 30% of the data for testing.
- **Stratified**: Ensures each class is represented proportionally in both train and test sets.

## Model Selection

- **CategoricalNB**: This is a variant of Naive Bayes specialized for categorical features.
- **Reasoning**: Efficient, straightforward, and well-suited for discrete inputs.

## Model Training

- **Fitting**: Train the CategoricalNB model on the training data.

## Evaluation

- **Predictions**: Generate predictions on the test set.
- **Accuracy**: Compute `accuracy_score` between the predicted labels and the true labels.
- **Typical Accuracy**: You can expect around 85-90% accuracy (and sometimes more) with this approach, depending on the random split.

# Streamlit Application

## How it Works

### User Input

- The user selects values for the six categorical features in the app’s sidebar or main page (depending on how you structure your UI).

### Model Inference

- The input is transformed using the same encoder that was used during training.
- The trained Naive Bayes model predicts a numeric code.
- That numeric code is reverse-mapped to the original category (e.g., "unacc", "acc", "good", "vgood").

### Prediction Output

- The predicted vehicle quality is displayed on the screen.

