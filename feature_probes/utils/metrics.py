"""Metrics computation utilities for probe evaluation."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


def compute_clf_metrics(
    preds: np.ndarray,
    labels: np.ndarray, 
    probs: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        preds: Binary predictions (0 or 1)
        labels: Ground truth labels (0 or 1)
        probs: Probability scores
        
    Returns:
        Dictionary of metrics
    """
    assert all((labels == 0.0) | (labels == 1.0)), "labels must be either 0 or 1"
    
    # Basic classification metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc_score = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float('nan')
    
    # Find optimal threshold
    optimal_threshold = float('nan')
    threshold_optimized_accuracy = float('nan')
    recall_at_01_fpr = float('nan')
    
    if len(np.unique(labels)) == 2:
        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        # Find optimal threshold for accuracy
        unique_probs = np.unique(probs)
        if len(unique_probs) > 100:
            percentiles = np.linspace(0, 100, 100)
            threshold_candidates = np.percentile(unique_probs, percentiles)
        else:
            threshold_candidates = unique_probs
        
        best_accuracy = 0.0
        optimal_threshold = 0.5
        
        for threshold in threshold_candidates:
            y_pred = (probs >= threshold).astype(int)
            acc = accuracy_score(labels, y_pred)
            if acc > best_accuracy:
                best_accuracy = acc
                optimal_threshold = threshold
        
        threshold_optimized_accuracy = best_accuracy
        
        # Calculate recall at 0.1 FPR
        target_fpr = 0.1
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            recall_at_01_fpr = tpr[idx[-1]]
        else:
            recall_at_01_fpr = 0.0
    
    # Count distributions
    true_positive_count = int(np.sum(labels == 1.0))
    true_negative_count = int(np.sum(labels == 0.0))
    pred_positive_count = int(np.sum(preds == 1.0))
    pred_negative_count = int(np.sum(preds == 0.0))
    total_samples = len(labels)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_score),
        "optimal_threshold": float(optimal_threshold),
        "threshold_optimized_accuracy": float(threshold_optimized_accuracy),
        "recall_at_0.1_fpr": float(recall_at_01_fpr),
        "true_positive_count": true_positive_count,
        "true_negative_count": true_negative_count,
        "pred_positive_count": pred_positive_count,
        "pred_negative_count": pred_negative_count,
        "total_samples": total_samples
    }


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        probabilities: Optional probability scores
        
    Returns:
        Dictionary of metrics
    """
    if probabilities is None:
        probabilities = predictions
    
    return compute_clf_metrics(predictions, labels, probabilities)


def compute_span_level_metrics(
    predictions: List[float],
    labels: List[float], 
    spans: List[List[int]]
) -> Dict[str, float]:
    """
    Compute span-level metrics by aggregating predictions over spans.
    
    Args:
        predictions: Token-level predictions
        labels: Span-level labels
        spans: List of [start, end] indices for each span
        
    Returns:
        Dictionary of span-level metrics
    """
    # Aggregate predictions by taking max over each span
    span_preds = []
    for span_indices in spans:
        if len(span_indices) == 2:
            start, end = span_indices
            span_pred = max(predictions[start:end+1])
            span_preds.append(span_pred)
    
    span_preds = np.array(span_preds)
    span_labels = np.array(labels[:len(span_preds)])
    
    # Convert to binary predictions
    binary_preds = (span_preds > 0.5).astype(float)
    
    return compute_clf_metrics(binary_preds, span_labels, span_preds)


def plot_roc_curves(
    all_preds: Dict[str, List[float]],
    all_labels: Dict[str, List[float]], 
    all_probs: Dict[str, List[float]],
    save_dir: str,
    prefix: Optional[str] = None
) -> None:
    """
    Plot ROC curves for different aggregation levels.
    
    Args:
        all_preds: Predictions for each aggregation level
        all_labels: Labels for each aggregation level
        all_probs: Probabilities for each aggregation level
        save_dir: Directory to save plots
        prefix: Optional prefix for filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 6))
    
    fpr_targets = [0.05, 0.1, 0.2, 0.5]
    dot_color = "black"
    dot_size = 40
    
    for i, agg_level in enumerate(['all', 'span', 'span_max']):
        plt.subplot(1, 3, i+1)
        
        if agg_level not in all_labels or len(all_labels[agg_level]) == 0:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient data")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        labels = np.array(all_labels[agg_level])
        probs = np.array(all_probs[agg_level])
        
        if len(np.unique(labels)) < 2:
            plt.title(f"{agg_level.replace('_', ' ').title()}\nInsufficient label diversity")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            continue
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        plt.fill_between(fpr, tpr, color="#f9c97d", alpha=0.5)
        plt.plot(fpr, tpr, lw=2, color="black", label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'w--', lw=2, alpha=0.7)
        
        # Mark TPR at specific FPRs
        for fpr_target in fpr_targets:
            idx = np.argmin(np.abs(fpr - fpr_target))
            plt.scatter(fpr[idx], tpr[idx], s=dot_size, color=dot_color, zorder=5)
            plt.text(fpr[idx], tpr[idx]+0.03, f"{tpr[idx]:.4f}", fontsize=10, ha="center", color=dot_color)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{agg_level.replace('_', ' ').title()}")
        plt.legend(loc="lower right")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    filename = f"{prefix}_roc_curves.png" if prefix else "roc_curves.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {os.path.join(save_dir, filename)}")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, save_path: str) -> None:
    """Plot a single ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(
    probabilities: np.ndarray,
    labels: np.ndarray, 
    save_path: str
) -> None:
    """Plot metrics vs threshold."""
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        accuracies.append(accuracy_score(labels, preds))
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_eval_metrics(
    metrics: dict,
    metric_key_prefix: str = "",
    all_labels: Optional[dict] = None,
    include_random_baseline: bool = True,
    seed: int = 42,
) -> None:
    """
    Print evaluation metrics in a nicely formatted way.
    
    Args:
        metrics: Dictionary of metrics
        metric_key_prefix: Optional prefix for metric keys
        all_labels: Optional labels for computing baselines
        include_random_baseline: Whether to include random baseline
        seed: Random seed for baseline
    """
    if metric_key_prefix:
        print(f"\n===== Evaluation Metrics ({metric_key_prefix}) =====")
    else:
        print("\n===== Evaluation Metrics =====")
    
    prefix = metric_key_prefix + "/" if metric_key_prefix else ""
    
    # Print loss metrics if available
    if f'{prefix}lm_loss' in metrics:
        print("\nLoss Metrics:")
        print(f" - LM Loss:     {metrics.get(f'{prefix}lm_loss', 0):.4f}")
        print(f" - Probe Loss:  {metrics.get(f'{prefix}probe_loss', 0):.4f}")
        print(f" - Sparsity:    {metrics.get(f'{prefix}sparsity', 0):.4f}")
    
    # Print classification metrics for different aggregation levels
    for agg_level in ['all', 'span', 'span_max']:
        if f'{prefix}{agg_level}_accuracy' in metrics:
            print(f"\n{agg_level.replace('_', ' ').title()} - Classification Metrics:")
            print(f" - Accuracy:   {metrics[f'{prefix}{agg_level}_accuracy']:.4f}")
            print(f" - Precision:  {metrics[f'{prefix}{agg_level}_precision']:.4f}")
            print(f" - Recall:     {metrics[f'{prefix}{agg_level}_recall']:.4f}")
            print(f" - F1 Score:   {metrics[f'{prefix}{agg_level}_f1']:.4f}")
            
            if f'{prefix}{agg_level}_auc' in metrics:
                print(f" - AUC:        {metrics[f'{prefix}{agg_level}_auc']:.4f}")
            if f'{prefix}{agg_level}_recall_at_0.1_fpr' in metrics:
                print(f" - Recall @ 0.1 FPR: {metrics[f'{prefix}{agg_level}_recall_at_0.1_fpr']:.4f}")
            if f'{prefix}{agg_level}_threshold_optimized_accuracy' in metrics:
                print(f" - Optimized Accuracy: {metrics[f'{prefix}{agg_level}_threshold_optimized_accuracy']:.4f}")
                print(f"   (Optimal Threshold: {metrics[f'{prefix}{agg_level}_optimal_threshold']:.4f})")
            
            # Print baselines if labels provided
            if all_labels and agg_level in all_labels:
                labels = np.array(all_labels[agg_level])
                if len(labels) > 0:
                    # Majority class baseline
                    majority_class = 1 if np.sum(labels) >= len(labels) / 2 else 0
                    majority_baseline = accuracy_score(labels, np.full_like(labels, majority_class))
                    print(f"    (Majority baseline: {majority_baseline:.4f})")
    
    print("\n==============================\n")