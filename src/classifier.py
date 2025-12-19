"""
classifier.py

Trains a logistic regression classifier on tiled methylation features to distinguish ALS from control samples.

Requirements:
    pip install scikit-learn pandas numpy

Usage:
    from classifier import train_methylation_classifier

    results = train_methylation_classifier(X, y, feature_names, method='elastic_net')
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
import pandas as pd

def train_methylation_classifier(
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: list, 
        method: str='elastic_net'):
    """
    Train classifier on tiled methylation features.
    
    Args:
        X: Feature matrix (samples x tiles)
        y: Labels (0=Control, 1=ALS)
        feature_names: List of tile IDs
        method: 'lasso', 'ridge', 'elastic_net'
    
    Returns:
        dict with model, metrics, and important features
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Set up regularization
    if method == 'lasso':
        l1_ratios = [1.0]  # Pure L1
        solver = 'saga'
    elif method == 'ridge':
        l1_ratios = [0.0]  # Pure L2
        solver = 'saga'  
    elif method == 'elastic_net':
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        solver = 'saga'

    # Cross-validated logistic regression
    model = LogisticRegressionCV(
        l1_ratios=l1_ratios,
        solver=solver,
        cv=5,
        scoring='roc_auc',
        max_iter=10000,
        random_state=42,
        use_legacy_attributes=False
    )
    
    model.fit(X_scaled, y)
    
    # Leave-one-out CV predictions (better for small samples)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
    y_prob = cross_val_predict(model, X_scaled, y, cv=loo, method='predict_proba')[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),  # TP / (TP + FP)
        'recall': recall_score(y, y_pred),        # Sensitivity: TP / (TP + FN)
        'sensitivity': recall_score(y, y_pred),   # Same as recall
        'specificity': recall_score(y, y_pred, pos_label=0),  # TN / (TN + FP)
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Extract important features
    coefs = model.coef_[0]
    feature_importance = pd.DataFrame({
        'tile': feature_names,
        'coefficient': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values('abs_coef', ascending=False)
    
    # Non-zero features (selected by L1)
    selected = feature_importance[feature_importance['coefficient'] != 0]
    
    results = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importance': feature_importance,
        'selected_features': selected,
        'n_features_selected': len(selected)
    }
    
    # Print results
    print(f"{'='*50}")
    print(f"CLASSIFICATION RESULTS ({method})")
    print(f"{'='*50}")
    print(f"\nSamples: {len(y)} (ALS: {sum(y)}, Control: {len(y)-sum(y)})")
    print(f"Features: {len(feature_names)}")
    print(f"Features selected: {results['n_features_selected']}")
    
    print(f"\n{'─'*50}")
    print("METRICS (Leave-One-Out Cross-Validation)")
    print(f"{'─'*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f} (Recall)")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    print(f"  F1 Score:    {metrics['f1']:.3f}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.3f}")
    
    print(f"\n{'─'*50}")
    print("CONFUSION MATRIX")
    print(f"{'─'*50}")
    print(f"                 Predicted")
    print(f"                 CTRL   ALS")
    print(f"  Actual CTRL    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"  Actual ALS     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    if len(selected) > 0:
        print(f"\n{'─'*50}")
        print("TOP SELECTED FEATURES")
        print(f"{'─'*50}")
        print(selected.head(10).to_string(index=False))
    
    return results