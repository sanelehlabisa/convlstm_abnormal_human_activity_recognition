"""Evaluation functions for model performance assessment.

Author: Sanele Hlabisa
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from typing import Optional

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    accuracy_function: nn.Module,
    other_metrics: list[nn.Module] = [],
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, dict[str, float]]:
    """
    Evaluate model on test data.

    Args:
        model: The neural network model to evaluate.
        data_loader: DataLoader for test data.
        loss_function: Loss function to compute loss.
        accuracy_function: Function to compute accuracy.
        other_metrics: List of additional metrics to compute (default: []).
        device: Device to run evaluation on (CPU or GPU).

    Returns:
        Tuple containing test loss and test accuracy.
    """
    loss, accuracy = 0.0, 0.0
    model = model.to(device)
    other_metrics_results = {}
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        for metric in other_metrics:
            metric.reset()
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits: torch.Tensor = model(X)
            probabilities = torch.softmax(logits, dim=1)
            class_indexes = probabilities.argmax(dim=1)
            batch_loss: torch.Tensor = loss_function(logits, y)
            loss += batch_loss.item()
            batch_accuracy: torch.Tensor = accuracy_function(class_indexes, y)
            accuracy += batch_accuracy.item()
            for metric in other_metrics:
                batch_metric: torch.Tensor = metric(class_indexes, y)
        for metric in other_metrics:
            other_metrics_results[metric.__class__.__name__] = metric.compute().item()
    print("Other Metrics:")
    for name, value in other_metrics_results.items():
        print(f"{name}: {value:.4f}")
    return (loss, accuracy, other_metrics_results)