#!/usr/bin/env python3
"""
Script to retrain the phishing URL detection model
This fixes the compatibility issue with newer scikit-learn versions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

def retrain_model():
    print("Loading dataset...")
    # Load the dataset
    data = pd.read_csv("phishing.csv")
    
    # Drop the index column
    data = data.drop(['Index'], axis=1)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns.tolist()}")
    
    # Separate features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train the model with the same parameters as the original
    print("Training GradientBoostingClassifier...")
    gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7, random_state=42)
    gbc.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gbc.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    print("\nSaving model to pickle/model.pkl...")
    with open("pickle/model.pkl", "wb") as file:
        pickle.dump(gbc, file)
    
    print("Model successfully retrained and saved!")
    
    # Test loading the model
    print("Testing model loading...")
    with open("pickle/model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    
    # Test prediction
    test_pred = loaded_model.predict(X_test[:1])
    print(f"Test prediction: {test_pred[0]}")
    print("Model loading successful!")

if __name__ == "__main__":
    retrain_model() 