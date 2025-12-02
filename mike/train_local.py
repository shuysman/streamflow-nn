#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local model training using AutoGluon (same as SageMaker Canvas/AutoPilot)

This script trains a model locally using the same AutoML approach that
SageMaker Canvas uses under the hood.

Usage:
    python train_local.py

Output:
    - Trained model saved to ./lamar_model/
    - Training metrics and leaderboard
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import datetime

def train_model():
    """Train AutoGluon model on Lamar River data"""

    print("="*60)
    print("LOCAL MODEL TRAINING (AutoGluon)")
    print("="*60)
    print()

    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv('Lamar_training_new.csv')

    print(f"Training data shape: {train_data.shape}")
    print(f"Features: {list(train_data.columns)}")
    print()

    # Define target variable (streamflow)
    label = 'cfslag7'  # This is the lagged streamflow we're predicting

    print(f"Target variable: {label}")
    print(f"Target statistics:")
    print(train_data[label].describe())
    print()

    # Remove any rows with missing target values
    train_data = train_data.dropna(subset=[label])
    print(f"Training samples after removing NaN targets: {len(train_data)}")
    print()

    # Create TabularDataset
    train_dataset = TabularDataset(train_data)

    # Configure predictor
    model_path = './lamar_model'

    print("Training model with AutoGluon...")
    print(f"Model will be saved to: {model_path}")
    print()
    print("Training configuration:")
    print("  - Problem type: regression")
    print("  - Time limit: 10 minutes (600 seconds)")
    print("  - Preset: medium_quality (balance speed/accuracy)")
    print("  - Eval metric: root_mean_squared_error")
    print()

    # Train model
    # This mimics what SageMaker Canvas does
    predictor = TabularPredictor(
        label=label,
        path=model_path,
        problem_type='regression',
        eval_metric='root_mean_squared_error'
    ).fit(
        train_data=train_dataset,
        time_limit=600,  # 10 minutes training time
        presets='medium_quality',  # Use medium quality for speed
        # Alternative presets:
        # - 'best_quality': Slower but more accurate
        # - 'high_quality': Good balance
        # - 'medium_quality': Faster training
        # - 'optimize_for_deployment': Faster inference
    )

    print()
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print()

    # Show model performance
    print("MODEL LEADERBOARD:")
    print("-"*60)
    leaderboard = predictor.leaderboard(train_dataset, silent=True)
    print(leaderboard)
    print()

    # Show feature importance
    print("FEATURE IMPORTANCE:")
    print("-"*60)
    feature_importance = predictor.feature_importance(train_dataset)
    print(feature_importance.head(10))
    print()

    # Get best model info
    best_model = predictor.get_model_best()
    print(f"Best model: {best_model}")
    print()

    # Test prediction on a sample
    print("Testing prediction on first sample...")
    sample = train_data.drop(columns=[label]).iloc[0:1]
    prediction = predictor.predict(sample)
    actual = train_data[label].iloc[0]
    print(f"  Predicted: {prediction.values[0]:.2f}")
    print(f"  Actual: {actual:.2f}")
    print()

    print("="*60)
    print("Model training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("Use this model with invoke_local.py for forecasting")
    print("="*60)

    return predictor

if __name__ == '__main__':
    # Check if training data exists
    if not os.path.exists('Lamar_training_new.csv'):
        print("ERROR: Lamar_training_new.csv not found!")
        print("Please run data_prep.py first to prepare the training data.")
        exit(1)

    # Train the model
    predictor = train_model()
