"""
PyTorch Deep Neural Network (DNN) Intrusion Detection Model
Designed for the 78 continuous numerical features of CICIDS2017
Classifies across 15 intrusion/benign classes.
Includes parameter flattening, differential gradient extraction,
and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from core.dataset import NUM_FEATURES, NUM_CLASSES, CICIDS_CLASSES

class IntrusionDetectionDNN(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, num_classes: int = NUM_CLASSES):
        super(IntrusionDetectionDNN, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch of size 1 gracefully with batch norm (during evaluation/inference)
        if x.size(0) == 1 and self.training:
            self.eval()
            out = self.net(x)
            self.train()
            return out
        return self.net(x)

    def get_flat_parameters(self) -> np.ndarray:
        """Returns all model parameters flattened into a 1D NumPy vector"""
        params = []
        for param in self.parameters():
            params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flat_parameters(self, flat_params: np.ndarray):
        """Loads model parameters from a flattened 1D NumPy vector"""
        offset = 0
        with torch.no_grad():
            for param in self.parameters():
                numel = param.numel()
                chunk = flat_params[offset:offset + numel]
                param.copy_(torch.tensor(chunk.reshape(param.shape), dtype=param.dtype))
                offset += numel

    def get_total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

def evaluate_model(
    model: IntrusionDetectionDNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Evaluates model performance: Loss, Accuracy, Precision, Recall, F1, and Confusion Matrix.
    """
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = total_loss / max(1, total_samples)

    acc = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    # Per-class attack detection rate
    class_acc = {}
    present_labels = sorted(list(set(all_targets)))
    for c_idx in present_labels:
        mask = np.array(all_targets) == c_idx
        if np.sum(mask) > 0:
            c_preds = np.array(all_preds)[mask]
            class_acc[CICIDS_CLASSES[c_idx]] = float(np.mean(c_preds == c_idx))

    return {
        "loss": float(round(avg_loss, 4)),
        "accuracy": float(round(acc * 100.0, 2)),
        "precision": float(round(float(prec) * 100.0, 2)),
        "recall": float(round(float(rec) * 100.0, 2)),
        "f1_score": float(round(float(f1) * 100.0, 2)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": class_acc,
        "sample_count": total_samples
    }
