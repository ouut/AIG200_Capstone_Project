"""
FashionMNIST Model Training: DNN vs CNN
========================================
This script performs:
1. Data loading and exploratory analysis
2. Preprocessing pipeline
3. DNN model training
4. CNN model training
5. Model comparison and export of the best model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import joblib
import json
import os
from datetime import datetime

# ─────────────────────────────────────
# Configuration
# ─────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
MODEL_DIR = "./models"
OUTPUT_DIR = "./output"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ─────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────

class FashionMNISTPreprocessor:
    """
    Preprocessing pipeline for FashionMNIST.
    Saves normalization stats for reuse at inference time.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, data_loader):
        """Compute mean and std from training data."""
        mean = 0.0
        std = 0.0
        total_samples = 0

        for images, _ in data_loader:
            batch_samples = images.size(0)
            # images shape: (batch, 1, 28, 28)
            mean += images.mean(dim=[0, 2, 3]) * batch_samples
            std += images.std(dim=[0, 2, 3]) * batch_samples
            total_samples += batch_samples

        self.mean = (mean / total_samples).item()
        self.std = (std / total_samples).item()
        self.is_fitted = True
        print(f"Fitted preprocessing: mean={self.mean:.4f}, std={self.std:.4f}")
        return self

    def transform(self):
        """Return torchvision transform based on fitted stats."""
        if not self.is_fitted:
            # Default: normalize to [0,1]
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])

    def save(self, path):
        joblib.dump({"mean": self.mean, "std": self.std, "is_fitted": self.is_fitted}, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path):
        data = joblib.load(path)
        preprocessor = FashionMNISTPreprocessor()
        preprocessor.mean = data["mean"]
        preprocessor.std = data["std"]
        preprocessor.is_fitted = data["is_fitted"]
        print(f"Preprocessor loaded from {path}")
        return preprocessor

    def preprocess_single(self, image_array):
        """
        Preprocess a single input image for inference.
        image_array: numpy array shape (28, 28) or (1, 28, 28), values 0-255 or 0-1
        Returns: torch tensor shape (1, 1, 28, 28)
        """
        # Convert to float and normalize to [0,1] if needed
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0

        # Ensure correct shape
        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, :, :]  # (1, 28, 28)
        elif image_array.ndim == 3 and image_array.shape[0] != 1:
            image_array = image_array[np.newaxis, :, :]  # Add channel dim

        tensor = torch.tensor(image_array, dtype=torch.float32)

        # Apply normalization
        if self.is_fitted and self.mean is not None:
            tensor = (tensor - self.mean) / self.std
        else:
            tensor = (tensor - 0.5) / 0.5

        # Ensure shape is (1, 1, 28, 28)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Fit preprocessor on raw data
raw_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor()
)
raw_train_loader = DataLoader(raw_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
preprocessor = FashionMNISTPreprocessor().fit(raw_train_loader)

# ─────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")
print(f"Classes: {len(class_names)}")

# Class distribution
train_labels = train_dataset.targets.numpy()
class_counts = np.bincount(train_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution plot
axes[0].bar(class_names, class_counts, color='steelblue')
axes[0].set_title("Training Set Class Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontsize=8)

# Sample images
for i in range(25):
    row, col = i // 5, i % 5
    img = train_dataset.data[i].numpy()
    label = train_dataset.targets[i].item()

    if len(axes) == 2:
        # Create a new subplot grid for samples
        pass

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

# Sample images grid
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i in range(25):
    row, col = i // 5, i % 5
    img = train_dataset.data[i].numpy()
    label = train_dataset.targets[i].item()
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(class_names[label], fontsize=9)
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=150)
plt.close()
print("EDA plots saved to output/")

# ─────────────────────────────────────
# 3. Model Definitions
# ─────────────────────────────────────

class DNN(nn.Module):
    """Deep Neural Network (fully-connected layers)."""

    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10, dropout=0.3):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class CNN(nn.Module):
    """Convolutional Neural Network for FashionMNIST."""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Block 1: (1, 28, 28) → (32, 14, 14)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2: (32, 14, 14) → (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3: (64, 7, 7) → (128, 3, 3)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ─────────────────────────────────────
# 4. Training & Evaluation Utilities
# ─────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def train_model(model, train_loader, test_loader, epochs, lr, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"Best test accuracy: {best_acc:.2f}%")
    return history


# ─────────────────────────────────────
# 5. Train Both Models
# ─────────────────────────────────────

print(f"\nModel input size: 28×28 = 784 features")
print(f"Training on: {DEVICE}")

# --- DNN ---
dnn_model = DNN(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    num_classes=10,
    dropout=0.3
).to(DEVICE)

dnn_history = train_model(
    dnn_model, train_loader, test_loader,
    epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, model_name="DNN"
)

# --- CNN ---
cnn_model = CNN(num_classes=10).to(DEVICE)

cnn_history = train_model(
    cnn_model, train_loader, test_loader,
    epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, model_name="CNN"
)

# ─────────────────────────────────────
# 6. Model Comparison
# ─────────────────────────────────────

print(f"\n{'='*50}")
print("Model Comparison")
print(f"{'='*50}")

# Get final metrics
_, dnn_train_acc, _, _ = evaluate(dnn_model, train_loader, nn.CrossEntropyLoss(), DEVICE)
dnn_test_loss, dnn_test_acc, dnn_preds, dnn_labels = evaluate(dnn_model, test_loader, nn.CrossEntropyLoss(), DEVICE)

_, cnn_train_acc, _, _ = evaluate(cnn_model, train_loader, nn.CrossEntropyLoss(), DEVICE)
cnn_test_loss, cnn_test_acc, cnn_preds, cnn_labels = evaluate(cnn_model, test_loader, nn.CrossEntropyLoss(), DEVICE)

# Count parameters
dnn_params = sum(p.numel() for p in dnn_model.parameters())
cnn_params = sum(p.numel() for p in cnn_model.parameters())

print(f"\n{'Metric':<25} {'DNN':>12} {'CNN':>12}")
print(f"{'-'*50}")
print(f"{'Train Accuracy':<25} {dnn_train_acc:>11.2f}% {cnn_train_acc:>11.2f}%")
print(f"{'Test Accuracy':<25} {dnn_test_acc:>11.2f}% {cnn_test_acc:>11.2f}%")
print(f"{'Test Loss':<25} {dnn_test_loss:>12.4f} {cnn_test_loss:>12.4f}")
print(f"{'Parameters':<25} {dnn_params:>12,} {cnn_params:>12,}")

# ─────────────────────────────────────
# 7. Classification Reports
# ─────────────────────────────────────

print(f"\n--- DNN Classification Report ---")
print(classification_report(dnn_labels, dnn_preds, target_names=class_names, digits=3))

print(f"\n--- CNN Classification Report ---")
print(classification_report(cnn_labels, cnn_preds, target_names=class_names, digits=3))

# ─────────────────────────────────────
# 8. Training Curves Plot
# ─────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(dnn_history["train_loss"], label="DNN Train", marker='o')
axes[0, 0].plot(dnn_history["test_loss"], label="DNN Test", marker='s')
axes[0, 0].plot(cnn_history["train_loss"], label="CNN Train", marker='o')
axes[0, 0].plot(cnn_history["test_loss"], label="CNN Test", marker='s')
axes[0, 0].set_title("Loss Curves")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(dnn_history["train_acc"], label="DNN Train", marker='o')
axes[0, 1].plot(dnn_history["test_acc"], label="DNN Test", marker='s')
axes[0, 1].plot(cnn_history["train_acc"], label="CNN Train", marker='o')
axes[0, 1].plot(cnn_history["test_acc"], label="CNN Test", marker='s')
axes[0, 1].set_title("Accuracy Curves")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# DNN Confusion Matrix
cm_dnn = confusion_matrix(dnn_labels, dnn_preds)
sns.heatmap(cm_dnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
axes[1, 0].set_title(f"DNN Confusion Matrix (Acc: {dnn_test_acc:.2f}%)")
axes[1, 0].set_xlabel("Predicted")
axes[1, 0].set_ylabel("True")
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].tick_params(axis='y', rotation=45)

# CNN Confusion Matrix
cm_cnn = confusion_matrix(cnn_labels, cnn_preds)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
axes[1, 1].set_title(f"CNN Confusion Matrix (Acc: {cnn_test_acc:.2f}%)")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("True")
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].tick_params(axis='y', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("\nComparison plots saved to output/model_comparison.png")

# ─────────────────────────────────────
# 9. Select and Export Best Model
# ─────────────────────────────────────

# Select best based on test accuracy
if cnn_test_acc >= dnn_test_acc:
    best_model = cnn_model
    best_model_type = "CNN"
    best_accuracy = cnn_test_acc
else:
    best_model = dnn_model
    best_model_type = "DNN"
    best_accuracy = dnn_test_acc

print(f"\nBest model: {best_model_type} (Test Accuracy: {best_accuracy:.2f}%)")

# Save model (TorchScript for production)
model_path = os.path.join(MODEL_DIR, "best_model.pt")
example_input = torch.randn(1, 1, 28, 28).to(DEVICE)

# Save state dict
torch.save({
    "model_type": best_model_type,
    "state_dict": best_model.state_dict(),
    "accuracy": best_accuracy,
    "class_names": class_names,
    "input_shape": [1, 28, 28],
    "num_classes": 10,
}, os.path.join(MODEL_DIR, "best_model_state.pt"))
print(f"Model state dict saved to {MODEL_DIR}/best_model_state.pt")

# Also save as TorchScript for optimized inference
best_model.eval()
with torch.no_grad():
    traced_model = torch.jit.trace(best_model.cpu(), example_input.cpu())
traced_model.save(model_path)
print(f"TorchScript model saved to {model_path}")

# Save preprocessing pipeline
preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.joblib"))

# Save model metadata
metadata = {
    "model_type": best_model_type,
    "framework": "PyTorch",
    "torch_version": torch.__version__,
    "accuracy": best_accuracy,
    "class_names": class_names,
    "input_shape": [1, 28, 28],
    "num_classes": 10,
    "preprocessing": {
        "mean": preprocessor.mean,
        "std": preprocessor.std,
    },
    "architecture": {
        "DNN": {
            "input_size": 784,
            "hidden_sizes": [512, 256, 128],
            "dropout": 0.3,
            "parameters": dnn_params,
            "test_accuracy": dnn_test_acc,
        },
        "CNN": {
            "parameters": cnn_params,
            "test_accuracy": cnn_test_acc,
        },
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "date": datetime.now().isoformat(),
    },
    "comparison": {
        "dnn_test_accuracy": dnn_test_acc,
        "cnn_test_accuracy": cnn_test_acc,
        "best_model": best_model_type,
    },
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to {MODEL_DIR}/metadata.json")

# ─────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────

print(f"\n{'='*50}")
print("Training Complete — Summary")
print(f"{'='*50}")
print(f"Best Model: {best_model_type}")
print(f"Test Accuracy: {best_accuracy:.2f}%")
print(f"Model saved at: {MODEL_DIR}/best_model.pt")
print(f"Preprocessor saved at: {MODEL_DIR}/preprocessor.joblib")
print(f"Metadata saved at: {MODEL_DIR}/metadata.json")
print(f"\nFiles ready for deployment!")
