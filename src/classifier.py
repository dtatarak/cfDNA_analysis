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
    
    # Leave-one-out CV predictions
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



def train_random_forest_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 3
) -> dict:
    """
    Train Random Forest classifier on tiled methylation features.
    
    Args:
        X: Feature matrix (samples x tiles)
        y: Labels (0=Control, 1=ALS)
        feature_names: List of tile IDs
        n_estimators: Number of trees
        max_depth: Maximum tree depth (limits overfitting)
        min_samples_leaf: Minimum samples per leaf
    
    Returns:
        dict with model, metrics, and important features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    # Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=1313
    )
    
    # Leave-one-out CV
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    y_prob = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'sensitivity': recall_score(y, y_pred),
        'specificity': recall_score(y, y_pred, pos_label=0),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob)
    }
    
    cm = confusion_matrix(y, y_pred)
    
    # Fit on full data for feature importance
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'tile': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results = {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importance': feature_importance
    }
    
    # Print results
    print("="*50)
    print("RANDOM FOREST RESULTS")
    print("="*50)
    print(f"\nSamples: {len(y)} (ALS: {sum(y)}, Control: {len(y)-sum(y)})")
    print(f"Features: {X.shape[1]}")
    
    print(f"\n{'─'*50}")
    print("METRICS (Leave-One-Out Cross-Validation)")
    print(f"{'─'*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
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
    
    print(f"\n{'─'*50}")
    print("TOP 10 IMPORTANT FEATURES")
    print(f"{'─'*50}")
    print(feature_importance.head(10).to_string(index=False))
    
    return results




import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from collections import defaultdict
import tempfile
from src.cfdna_methylkit import run_methylkit_tiled_dmr_analysis

def bootstrap_dmr_classifier(
    coverage_files: list,
    sample_ids: list,
    labels: list,
    n_iterations: int = 20,
    n_holdout_per_class: int = 3,
    tile_window: int = 5000,
    tile_step: int = 5000,
    tile_min_coverage: int = 5,
    min_diff: float = 0.1,
    qvalue: float = 0.05,
    classifier: str = 'logistic',
    method: str = 'lasso',
    verbose: bool = True
):
    """
    Bootstrap validation of DMR-based classification with LOO-CV.
    
    For each iteration:
    1. Hold out n samples per class (not used for DMR discovery)
    2. Run DMR analysis on remaining samples
    3. Do LOO-CV on ALL samples using discovered DMRs
    
    Args:
        coverage_files: List of paths to Bismark coverage files
        sample_ids: List of sample IDs
        labels: List of labels (0=Control, 1=ALS)
        n_iterations: Number of bootstrap iterations
        n_holdout_per_class: Number of samples per class to hold out from DMR discovery
        tile_window: Window size for tiling
        tile_min_coverage: Minimum coverage per tile
        min_diff: Minimum methylation difference for DMR
        qvalue: Q-value threshold for DMR significance
        classifier: 'random_forest' or 'logistic'
        method: 'lasso', 'ridge', 'elastic_net' (for logistic regression)
        verbose: Print progress
    
    Returns:
        dict with metrics, DMR stability, feature importance, and selected features
    """
    labels = np.array(labels)
    n_samples = len(labels)
    
    # Get indices for each class
    ctrl_idx = np.where(labels == 0)[0]
    als_idx = np.where(labels == 1)[0]
    
    print(f"Total samples: {n_samples} (ALS: {len(als_idx)}, Control: {len(ctrl_idx)})")
    print(f"Holding out {n_holdout_per_class} per class from DMR discovery")
    print(f"LOO-CV on all {n_samples} samples for classification")
    
    # Storage for results
    all_metrics = defaultdict(list)
    all_dmrs = []
    all_importances = defaultdict(list)
    all_predictions = []
    all_selected_features = []
    
    np.random.seed(42)
    
    for fold in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {fold + 1}/{n_iterations}")
            print(f"{'='*60}")
        
        # Randomly select samples to hold out from DMR discovery
        holdout_ctrl = np.random.choice(ctrl_idx, size=n_holdout_per_class, replace=False)
        holdout_als = np.random.choice(als_idx, size=n_holdout_per_class, replace=False)
        holdout_idx = np.concatenate([holdout_ctrl, holdout_als])
        dmr_idx = np.array([i for i in range(n_samples) if i not in holdout_idx])
        
        if verbose:
            print(f"DMR discovery: {len(dmr_idx)} samples")
            print(f"Held out from DMR: {len(holdout_idx)} samples")
        
        # Get files for DMR discovery
        dmr_files = [coverage_files[i] for i in dmr_idx]
        dmr_ids = [sample_ids[i] for i in dmr_idx]
        dmr_labels = labels[dmr_idx]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # Step 1: Run DMR analysis (excluding held-out samples)
            if verbose:
                print(f"\nStep 1: Finding DMRs...")
            
            try:
                dmr_results = run_methylkit_tiled_dmr_analysis(
                    coverage_files=dmr_files,
                    sample_ids=dmr_ids,
                    treatments=list(dmr_labels),
                    output_dir=temp_dir,
                    tile_window=tile_window,
                    tile_step=tile_step,
                    tile_min_coverage=tile_min_coverage,
                    min_diff=min_diff,
                    min_per_group=1,
                    qvalue=qvalue,
                    verbose=False
                )
            except Exception as e:
                print(f"  DMR analysis failed: {e}")
                continue
            
            sig_tiles = dmr_results['tiles_significant']
            
            if len(sig_tiles) == 0:
                if verbose:
                    print("  No significant DMRs found, skipping iteration")
                continue
            
            if verbose:
                print(f"  Found {len(sig_tiles)} significant DMRs")
            
            # Track DMRs found
            dmr_tile_ids = [f"chr21_{int(row['start'])}_{int(row['end'])}" 
                           for _, row in sig_tiles.iterrows()]
            all_dmrs.append(set(dmr_tile_ids))
            
            # Step 2: Get tile matrix for ALL samples
            if verbose:
                print(f"\nStep 2: Extracting methylation for all samples...")

            full_results = run_methylkit_tiled_dmr_analysis(
                coverage_files=coverage_files,
                sample_ids=sample_ids,
                treatments=list(labels),
                output_dir=temp_dir,
                tile_window=tile_window,
                tile_min_coverage=1,
                min_diff=0,
                qvalue=1,
                verbose=False
            )

            full_tile_matrix = full_results.get('tile_matrix')

            if full_tile_matrix is None:
                if verbose:
                    print("  No tile matrix available, skipping iteration")
                continue

            # Subset to significant DMRs
            available_sig = [t for t in dmr_tile_ids if t in full_tile_matrix.columns]

            if len(available_sig) == 0:
                if verbose:
                    print("  No significant tiles in matrix, skipping iteration")
                continue

            # Keep only tiles with no missing values
            X_sig = full_tile_matrix[available_sig]
            complete_tiles = X_sig.columns[X_sig.isna().sum() == 0].tolist()

            if len(complete_tiles) == 0:
                if verbose:
                    print("  No complete tiles, skipping iteration")
                continue

            if verbose:
                print(f"  Significant tiles: {len(available_sig)}")
                print(f"  Complete tiles (no NaN): {len(complete_tiles)}")

            X = X_sig[complete_tiles].values
            y = labels
            available_sig = complete_tiles
            
            # Step 3: LOO-CV on all samples
            if verbose:
                print(f"\nStep 3: Running LOO-CV with {classifier}...")
            
            y_pred = np.zeros(n_samples, dtype=int)
            y_prob = np.zeros(n_samples)
            selected_features_this_fold = set()
            
            for i in range(n_samples):
                # Leave one out
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[i] = False
                
                X_train = X[train_mask]
                y_train = y[train_mask]
                X_test = X[i:i+1]
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train
                if classifier == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=42
                    )
                else:
                    # Set up regularization
                    if method == 'lasso':
                        l1_ratios = [1.0]
                        solver = 'saga'
                    elif method == 'ridge':
                        l1_ratios = [0.0]
                        solver = 'saga'
                    elif method == 'elastic_net':
                        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
                        solver = 'saga'

                    model = LogisticRegressionCV(
                        l1_ratios=l1_ratios,
                        solver=solver,
                        cv=5,
                        scoring='roc_auc',
                        max_iter=10000,
                        random_state=42,
                        use_legacy_attributes=False
                    )
                
                model.fit(X_train_scaled, y_train)
                
                # Track non-zero coefficients for logistic regression
                if classifier != 'random_forest':
                    nonzero_idx = np.where(model.coef_[0] != 0)[0]
                    for idx in nonzero_idx:
                        selected_features_this_fold.add(available_sig[idx])
                
                # Predict
                y_pred[i] = model.predict(X_test_scaled)[0]
                y_prob[i] = model.predict_proba(X_test_scaled)[0, 1]
            
            # Store selected features for this fold
            all_selected_features.append(selected_features_this_fold)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
                'f1': f1_score(y, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y, y_prob),
                'n_dmrs': len(available_sig),
                'n_features_selected': len(selected_features_this_fold)
            }
            
            for key, val in metrics.items():
                all_metrics[key].append(val)
            
            all_predictions.append({
                'y_true': y.copy(), 
                'y_pred': y_pred.copy(), 
                'y_prob': y_prob.copy(),
                'selected_features': list(selected_features_this_fold)
            })
            
            # Get feature importance from full model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if classifier == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=2, random_state=42)
                model.fit(X_scaled, y)
                for name, imp in zip(available_sig, model.feature_importances_):
                    all_importances[name].append(imp)
            else:
                # Set up regularization for final model
                if method == 'lasso':
                    l1_ratios = [1.0]
                elif method == 'ridge':
                    l1_ratios = [0.0]
                elif method == 'elastic_net':
                    l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
                
                model = LogisticRegressionCV(
                    l1_ratios=l1_ratios,
                    solver='saga',
                    cv=5,
                    scoring='roc_auc',
                    max_iter=10000,
                    random_state=42,
                    use_legacy_attributes=False
                )
                model.fit(X_scaled, y)
                for name, coef in zip(available_sig, model.coef_[0]):
                    all_importances[name].append(abs(coef))
            
            if verbose:
                cm = confusion_matrix(y, y_pred)
                print(f"\n  LOO-CV Results:")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    ROC AUC:  {metrics['roc_auc']:.3f}")
                print(f"    F1 Score: {metrics['f1']:.3f}")
                if classifier != 'random_forest':
                    print(f"    Features selected: {metrics['n_features_selected']}")
                print(f"    Confusion Matrix:")
                print(f"      CTRL: {cm[0,0]} correct, {cm[0,1]} misclassified")
                print(f"      ALS:  {cm[1,1]} correct, {cm[1,0]} misclassified")
    
    # Summarize results
    print("\n" + "="*60)
    print("BOOTSTRAP SUMMARY")
    print("="*60)
    
    print(f"\nSuccessful iterations: {len(all_metrics['accuracy'])}/{n_iterations}")
    
    print(f"\n{'─'*60}")
    print("PERFORMANCE METRICS (mean ± std)")
    print(f"{'─'*60}")
    for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']:
        vals = all_metrics[metric]
        if len(vals) > 0:
            print(f"  {metric:<15}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    
    print(f"\n{'─'*60}")
    print("DMR STATISTICS")
    print(f"{'─'*60}")
    n_dmrs = all_metrics['n_dmrs']
    if len(n_dmrs) > 0:
        print(f"  DMRs per iteration: {np.mean(n_dmrs):.1f} ± {np.std(n_dmrs):.1f}")
    
    # DMR stability
    stable_dmrs = {}
    if len(all_dmrs) > 1:
        all_dmr_set = set().union(*all_dmrs)
        dmr_counts = {dmr: sum(1 for s in all_dmrs if dmr in s) for dmr in all_dmr_set}
        stable_dmrs = {k: v for k, v in dmr_counts.items() if v >= len(all_dmrs) * 0.5}
        
        print(f"  Unique DMRs found: {len(all_dmr_set)}")
        print(f"  Stable DMRs (≥50% iterations): {len(stable_dmrs)}")
        
        if len(stable_dmrs) > 0:
            print(f"\n  Most stable DMRs:")
            sorted_dmrs = sorted(dmr_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for dmr, count in sorted_dmrs:
                print(f"    {dmr}: {count}/{len(all_dmrs)} iterations ({count/len(all_dmrs)*100:.0f}%)")
    
    # Feature selection summary (for logistic regression)
    consistent_features = {}
    if len(all_selected_features) > 0 and classifier != 'random_forest':
        print(f"\n{'─'*60}")
        print("FEATURE SELECTION (Logistic Regression)")
        print(f"{'─'*60}")
        
        n_selected = all_metrics.get('n_features_selected', [])
        if len(n_selected) > 0:
            print(f"  Features selected per iteration: {np.mean(n_selected):.1f} ± {np.std(n_selected):.1f}")
        
        # Find consistently selected features
        all_selected_set = set().union(*all_selected_features) if all_selected_features else set()
        feature_selection_counts = {f: sum(1 for s in all_selected_features if f in s) for f in all_selected_set}
        consistent_features = {k: v for k, v in feature_selection_counts.items() if v >= len(all_selected_features) * 0.5}
        
        print(f"  Unique features ever selected: {len(all_selected_set)}")
        print(f"  Consistent features (≥50% iterations): {len(consistent_features)}")
        
        if len(feature_selection_counts) > 0:
            print(f"\n  Most consistently selected features:")
            sorted_features = sorted(feature_selection_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, count in sorted_features:
                print(f"    {feat}: {count}/{len(all_selected_features)} iterations ({count/len(all_selected_features)*100:.0f}%)")
    
    print(f"\n{'─'*60}")
    print("MOST IMPORTANT FEATURES")
    print(f"{'─'*60}")
    
    mean_importance = {k: np.mean(v) for k, v in all_importances.items() if len(v) > 0}
    top_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for tile, imp in top_features:
        n_appearances = len(all_importances[tile])
        print(f"  {tile}: importance={imp:.4f} (appeared {n_appearances}x)")
    
    return {
        'metrics': dict(all_metrics),
        'predictions': all_predictions,
        'dmrs_per_fold': all_dmrs,
        'stable_dmrs': stable_dmrs,
        'feature_importance': dict(all_importances),
        'selected_features_per_fold': all_selected_features,
        'consistent_features': consistent_features,
        'summary': {
            'mean_accuracy': np.mean(all_metrics['accuracy']) if all_metrics['accuracy'] else 0,
            'std_accuracy': np.std(all_metrics['accuracy']) if all_metrics['accuracy'] else 0,
            'mean_auc': np.mean(all_metrics['roc_auc']) if all_metrics['roc_auc'] else 0,
            'std_auc': np.std(all_metrics['roc_auc']) if all_metrics['roc_auc'] else 0,
            'mean_f1': np.mean(all_metrics['f1']) if all_metrics['f1'] else 0,
            'std_f1': np.std(all_metrics['f1']) if all_metrics['f1'] else 0,
            'n_stable_dmrs': len(stable_dmrs),
            'mean_features_selected': np.mean(all_metrics.get('n_features_selected', [0])),
            'n_consistent_features': len(consistent_features)
        }
    }



def run_bootstrap_analysis(
    coverage_files: list,
    sample_ids: list,
    labels: list,
    n_iterations: int = 20,
    n_holdout_per_class: int = 3,
    tile_window: int = 5000,
    tile_step: int = 5000,
    classifier: str = 'logistic',
    method: str = 'lasso',
    verbose: bool = False
):
    """
    Run bootstrap DMR classifier and print comprehensive report.
    
    Args:
        coverage_files: List of paths to Bismark coverage files
        sample_ids: List of sample IDs
        labels: List of labels (0=Control, 1=ALS)
        n_iterations: Number of bootstrap iterations
        n_holdout_per_class: Number of samples per class to hold out
        tile_window: Window size for tiling
        classifier: 'random_forest' or 'logistic'
        method: 'lasso', 'ridge', 'elastic_net' (for logistic)
        verbose: Print progress during bootstrapping
    
    Returns:
        dict with all results
    """
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("BOOTSTRAP DMR CLASSIFICATION ANALYSIS")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Iterations: {n_iterations}")
    print(f"  Holdout per class: {n_holdout_per_class}")
    print(f"  Tile window: {tile_window}bp")
    print(f"  Classifier: {classifier}")
    if classifier == 'logistic':
        print(f"  Regularization: {method}")
    print()
    
    # Run bootstrap
    results = bootstrap_dmr_classifier(
        coverage_files=coverage_files,
        sample_ids=sample_ids,
        labels=labels,
        n_iterations=n_iterations,
        n_holdout_per_class=n_holdout_per_class,
        tile_window=tile_window,
        tile_step=tile_step,
        classifier=classifier,
        method=method,
        verbose=verbose
    )
    
    # Print detailed report
    print("\n")
    print("="*70)
    print("DETAILED STABILITY REPORT")
    print("="*70)
    
    # Performance consistency
    print(f"\n{'─'*70}")
    print("1. PERFORMANCE CONSISTENCY")
    print(f"{'─'*70}")
    
    metrics = results['metrics']
    
    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'CV%':>8}")
    print(f"  {'-'*55}")
    
    for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc']:
        vals = metrics.get(metric, [])
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            min_val = np.min(vals)
            max_val = np.max(vals)
            cv = (std_val / mean_val * 100) if mean_val > 0 else 0
            print(f"  {metric:<15} {mean_val:>8.3f} {std_val:>8.3f} {min_val:>8.3f} {max_val:>8.3f} {cv:>7.1f}%")
    
    # Performance interpretation
    mean_auc = results['summary']['mean_auc']
    std_auc = results['summary']['std_auc']
    
    print(f"\n  Interpretation:")
    if mean_auc >= 0.8:
        print(f"    ✓ Good discrimination (AUC = {mean_auc:.3f})")
    elif mean_auc >= 0.6:
        print(f"    ~ Moderate discrimination (AUC = {mean_auc:.3f})")
    else:
        print(f"    ✗ Poor discrimination (AUC = {mean_auc:.3f})")
    
    if std_auc < 0.1:
        print(f"    ✓ Stable performance (std = {std_auc:.3f})")
    elif std_auc < 0.2:
        print(f"    ~ Moderate variability (std = {std_auc:.3f})")
    else:
        print(f"    ✗ High variability (std = {std_auc:.3f})")
    
    # DMR consistency
    print(f"\n{'─'*70}")
    print("2. DMR CONSISTENCY ACROSS ITERATIONS")
    print(f"{'─'*70}")
    
    all_dmrs = results['dmrs_per_fold']
    n_successful = len(all_dmrs)
    
    if n_successful > 1:
        all_dmr_set = set().union(*all_dmrs)
        dmr_counts = {dmr: sum(1 for s in all_dmrs if dmr in s) for dmr in all_dmr_set}
        
        # Categorize DMRs by stability
        always_found = [d for d, c in dmr_counts.items() if c == n_successful]
        often_found = [d for d, c in dmr_counts.items() if c >= n_successful * 0.75 and c < n_successful]
        sometimes_found = [d for d, c in dmr_counts.items() if c >= n_successful * 0.5 and c < n_successful * 0.75]
        rarely_found = [d for d, c in dmr_counts.items() if c < n_successful * 0.5]
        
        print(f"\n  Total unique DMRs discovered: {len(all_dmr_set)}")
        print(f"  DMRs per iteration: {np.mean([len(d) for d in all_dmrs]):.1f} ± {np.std([len(d) for d in all_dmrs]):.1f}")
        print(f"\n  DMR Stability Breakdown:")
        print(f"    Found in 100% of iterations:  {len(always_found):>4} DMRs")
        print(f"    Found in 75-99% of iterations: {len(often_found):>4} DMRs")
        print(f"    Found in 50-74% of iterations: {len(sometimes_found):>4} DMRs")
        print(f"    Found in <50% of iterations:  {len(rarely_found):>4} DMRs")
        
        # Jaccard similarity between iterations
        jaccard_scores = []
        for i in range(len(all_dmrs)):
            for j in range(i+1, len(all_dmrs)):
                intersection = len(all_dmrs[i] & all_dmrs[j])
                union = len(all_dmrs[i] | all_dmrs[j])
                if union > 0:
                    jaccard_scores.append(intersection / union)
        
        if jaccard_scores:
            print(f"\n  DMR Set Similarity (Jaccard):")
            print(f"    Mean: {np.mean(jaccard_scores):.3f}")
            print(f"    Min:  {np.min(jaccard_scores):.3f}")
            print(f"    Max:  {np.max(jaccard_scores):.3f}")
            
            if np.mean(jaccard_scores) > 0.7:
                print(f"    ✓ High consistency - similar DMRs found across iterations")
            elif np.mean(jaccard_scores) > 0.4:
                print(f"    ~ Moderate consistency - some overlap in DMRs")
            else:
                print(f"    ✗ Low consistency - different DMRs found each iteration")
    
    # Feature selection consistency (for logistic regression)
    if classifier == 'logistic' and len(results['selected_features_per_fold']) > 0:
        print(f"\n{'─'*70}")
        print("3. FEATURE SELECTION CONSISTENCY")
        print(f"{'─'*70}")
        
        all_selected = results['selected_features_per_fold']
        n_selected_per_fold = [len(s) for s in all_selected]
        
        print(f"\n  Features selected per iteration: {np.mean(n_selected_per_fold):.1f} ± {np.std(n_selected_per_fold):.1f}")
        print(f"  Range: {np.min(n_selected_per_fold)} - {np.max(n_selected_per_fold)}")
        
        if len(all_selected) > 1:
            all_selected_set = set().union(*all_selected)
            feature_counts = {f: sum(1 for s in all_selected if f in s) for f in all_selected_set}
            
            always_selected = [f for f, c in feature_counts.items() if c == len(all_selected)]
            often_selected = [f for f, c in feature_counts.items() if c >= len(all_selected) * 0.75 and c < len(all_selected)]
            
            print(f"\n  Feature Selection Stability:")
            print(f"    Total unique features selected: {len(all_selected_set)}")
            print(f"    Selected in 100% of iterations: {len(always_selected)}")
            print(f"    Selected in 75-99% of iterations: {len(often_selected)}")
            
            # Feature selection Jaccard
            feature_jaccard = []
            for i in range(len(all_selected)):
                for j in range(i+1, len(all_selected)):
                    if len(all_selected[i]) > 0 and len(all_selected[j]) > 0:
                        intersection = len(all_selected[i] & all_selected[j])
                        union = len(all_selected[i] | all_selected[j])
                        if union > 0:
                            feature_jaccard.append(intersection / union)
            
            if feature_jaccard:
                print(f"\n  Feature Set Similarity (Jaccard):")
                print(f"    Mean: {np.mean(feature_jaccard):.3f}")
                
                if np.mean(feature_jaccard) > 0.7:
                    print(f"    ✓ Highly consistent feature selection")
                elif np.mean(feature_jaccard) > 0.4:
                    print(f"    ~ Moderately consistent feature selection")
                else:
                    print(f"    ✗ Inconsistent feature selection")
    
    # Top features
    print(f"\n{'─'*70}")
    print("4. TOP PREDICTIVE FEATURES")
    print(f"{'─'*70}")
    
    mean_importance = {k: np.mean(v) for k, v in results['feature_importance'].items() if len(v) > 0}
    
    if mean_importance:
        top_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print(f"\n  {'Rank':<6} {'Tile':<30} {'Importance':>12} {'Appearances':>12}")
        print(f"  {'-'*62}")
        
        for rank, (tile, imp) in enumerate(top_features, 1):
            n_appearances = len(results['feature_importance'][tile])
            pct_appearances = n_appearances / n_successful * 100
            print(f"  {rank:<6} {tile:<30} {imp:>12.4f} {n_appearances:>4}/{n_successful} ({pct_appearances:>5.1f}%)")
    
    # Overall assessment
    print(f"\n{'─'*70}")
    print("5. OVERALL ASSESSMENT")
    print(f"{'─'*70}")
    
    # Calculate overall score
    scores = []
    
    # Performance score
    if mean_auc >= 0.8:
        scores.append(("Performance", "Good", 3))
    elif mean_auc >= 0.6:
        scores.append(("Performance", "Moderate", 2))
    else:
        scores.append(("Performance", "Poor", 1))
    
    # Stability score
    if std_auc < 0.1:
        scores.append(("Stability", "Good", 3))
    elif std_auc < 0.2:
        scores.append(("Stability", "Moderate", 2))
    else:
        scores.append(("Stability", "Poor", 1))
    
    # DMR consistency score
    if n_successful > 1 and jaccard_scores:
        if np.mean(jaccard_scores) > 0.7:
            scores.append(("DMR Consistency", "Good", 3))
        elif np.mean(jaccard_scores) > 0.4:
            scores.append(("DMR Consistency", "Moderate", 2))
        else:
            scores.append(("DMR Consistency", "Poor", 1))
    
    print(f"\n  {'Category':<20} {'Rating':<12}")
    print(f"  {'-'*32}")
    for category, rating, _ in scores:
        symbol = "✓" if rating == "Good" else ("~" if rating == "Moderate" else "✗")
        print(f"  {category:<20} {symbol} {rating:<10}")
    
    avg_score = np.mean([s[2] for s in scores])
    print(f"\n  Overall confidence: ", end="")
    if avg_score >= 2.5:
        print("HIGH - Results are likely reproducible")
    elif avg_score >= 1.5:
        print("MODERATE - Results should be validated")
    else:
        print("LOW - Results may not be reliable")
    
    return results
