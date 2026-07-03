#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FashionMNIST: DNN vs CNN — Model Training & Comparison
#
# **AIG200 Capstone — ML Deployment Assignment (Spring 2026)**
#
# This notebook covers:
# 1. Data loading & EDA
# 2. Preprocessing pipeline
# 3. DNN model training
# 4. CNN model training
# 5. Model comparison & selection
# 6. Export for deployment

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime

# Config
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

# %% [markdown]
# ## 2. Data Loading & Preprocessing Pipeline

# %%
class FashionMNISTPreprocessor:
    """Preprocessing pipeline for FashionMNIST — reusable at inference time."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, data_loader):
        mean = 0.0; std = 0.0; total_samples = 0
        for images, _ in data_loader:
            batch_samples = images.size(0)
            mean += images.mean(dim=[0, 2, 3]) * batch_samples
            std += images.std(dim=[0, 2, 3]) * batch_samples
            total_samples += batch_samples
        self.mean = (mean / total_samples).item()
        self.std = (std / total_samples).item()
        self.is_fitted = True
        print(f"Fitted: mean={self.mean:.4f}, std={self.std:.4f}")
        return self

    def transform(self):
        if not self.is_fitted:
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

    @staticmethod
    def load(path):
        data = joblib.load(path)
        p = FashionMNISTPreprocessor()
        p.mean, p.std, p.is_fitted = data["mean"], data["std"], data["is_fitted"]
        return p

    def preprocess_single(self, image_array):
        if image_array.max() > 1.0:
            image_array = image_array.astype(np.float32) / 255.0
        if image_array.ndim == 2:
            image_array = image_array[np.newaxis, :, :]
        tensor = torch.tensor(image_array, dtype=torch.float32)
        if self.is_fitted:
            tensor = (tensor - self.mean) / self.std
        else:
            tensor = (tensor - 0.5) / 0.5
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor


# Standard transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download & load
train_dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Fit preprocessor on raw pixel data
raw_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
preprocessor = FashionMNISTPreprocessor().fit(DataLoader(raw_train, batch_size=BATCH_SIZE, shuffle=False))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Classes: {len(class_names)}")

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
# Class distribution
train_labels = train_dataset.targets.numpy()
class_counts = np.bincount(train_labels)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_names, class_counts, color='steelblue', edgecolor='white')
ax.set_title("Training Set — Class Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Class"); ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(count), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
plt.show()

# %%
# Sample images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle("FashionMNIST — Sample Images", fontsize=14, fontweight='bold')
for i in range(25):
    row, col = i // 5, i % 5
    img = train_dataset.data[i].numpy()
    label = train_dataset.targets[i].item()
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(class_names[label], fontsize=9)
    axes[row, col].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"), dpi=150)
plt.show()
print("EDA plots saved!")

# %% [markdown]
# ## 4. Model Architectures

# %%
class DNN(nn.Module):
    """Deep Neural Network — fully connected layers with BatchNorm & Dropout."""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.3):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(self.flatten(x))


class CNN(nn.Module):
    """CNN — 3 Conv blocks + FC classifier."""

    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 2: 14x14 → 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            # Block 3: 7x7 → 3x3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# %% [markdown]
# ## 5. Training & Evaluation Helpers

# %%
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = model(images).max(1)
        total += labels.size(0); correct += preds.eq(labels).sum().item()
    return running_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0); correct += preds.eq(labels).sum().item()
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def train_model(model, train_ldr, test_ldr, epochs, lr, device, name):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    for ep in range(epochs):
        tl, ta = train_one_epoch(model, train_ldr, criterion, opt, device)
        vl, va, _, _ = evaluate(model, test_ldr, criterion, device)
        sched.step(vl)
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["test_loss"].append(vl); history["test_acc"].append(va)
        print(f"Epoch {ep+1:2d}/{epochs} | Train Loss: {tl:.4f} Acc: {ta:.2f}% | Test Loss: {vl:.4f} Acc: {va:.2f}%")
        if va > best_acc:
            best_acc = va; best_state = model.state_dict().copy()
    model.load_state_dict(best_state)
    print(f"Best test accuracy: {best_acc:.2f}%")
    return history

# %% [markdown]
# ## 6. Train DNN

# %%
dnn_model = DNN(input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.3).to(DEVICE)
dnn_params = sum(p.numel() for p in dnn_model.parameters())
print(f"DNN parameters: {dnn_params:,}")

dnn_history = train_model(dnn_model, train_loader, test_loader,
                          epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, name="DNN")

# %%
# DNN evaluation
_, dnn_train_acc, _, _ = evaluate(dnn_model, train_loader, nn.CrossEntropyLoss(), DEVICE)
dnn_loss, dnn_acc, dnn_preds, dnn_labels = evaluate(dnn_model, test_loader, nn.CrossEntropyLoss(), DEVICE)
print(f"\nDNN — Train Acc: {dnn_train_acc:.2f}% | Test Acc: {dnn_acc:.2f}%")
print(classification_report(dnn_labels, dnn_preds, target_names=class_names, digits=3))

# %% [markdown]
# ## 7. Train CNN

# %%
cnn_model = CNN(num_classes=10).to(DEVICE)
cnn_params = sum(p.numel() for p in cnn_model.parameters())
print(f"CNN parameters: {cnn_params:,}")

cnn_history = train_model(cnn_model, train_loader, test_loader,
                          epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, name="CNN")

# %%
# CNN evaluation
_, cnn_train_acc, _, _ = evaluate(cnn_model, train_loader, nn.CrossEntropyLoss(), DEVICE)
cnn_loss, cnn_acc, cnn_preds, cnn_labels = evaluate(cnn_model, test_loader, nn.CrossEntropyLoss(), DEVICE)
print(f"\nCNN — Train Acc: {cnn_train_acc:.2f}% | Test Acc: {cnn_acc:.2f}%")
print(classification_report(cnn_labels, cnn_preds, target_names=class_names, digits=3))

# %% [markdown]
# ## 8. Model Comparison

# %%
print(f"\n{'='*60}")
print(f"{'Metric':<25} {'DNN':>15} {'CNN':>15}")
print(f"{'='*60}")
print(f"{'Train Accuracy':<25} {dnn_train_acc:>14.2f}% {cnn_train_acc:>14.2f}%")
print(f"{'Test Accuracy':<25} {dnn_acc:>14.2f}% {cnn_acc:>14.2f}%")
print(f"{'Test Loss':<25} {dnn_loss:>15.4f} {cnn_loss:>15.4f}")
print(f"{'Parameters':<25} {dnn_params:>15,} {cnn_params:>15,}")

# %%
# Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
for hist, name, c in [(dnn_history, "DNN", "blue"), (cnn_history, "CNN", "red")]:
    axes[0, 0].plot(hist["train_loss"], 'o-', label=f"{name} Train", color=c, alpha=0.7)
    axes[0, 0].plot(hist["test_loss"], 's--', label=f"{name} Test", color=c, alpha=0.7)
axes[0, 0].set_title("Loss Curves", fontweight='bold'); axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# Accuracy
for hist, name, c in [(dnn_history, "DNN", "blue"), (cnn_history, "CNN", "red")]:
    axes[0, 1].plot(hist["train_acc"], 'o-', label=f"{name} Train", color=c, alpha=0.7)
    axes[0, 1].plot(hist["test_acc"], 's--', label=f"{name} Test", color=c, alpha=0.7)
axes[0, 1].set_title("Accuracy Curves", fontweight='bold'); axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

# Confusion matrices
sns.heatmap(confusion_matrix(dnn_labels, dnn_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
axes[1, 0].set_title(f"DNN (Acc: {dnn_acc:.1f}%)", fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45); axes[1, 0].tick_params(axis='y', rotation=45)

sns.heatmap(confusion_matrix(cnn_labels, cnn_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
axes[1, 1].set_title(f"CNN (Acc: {cnn_acc:.1f}%)", fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45); axes[1, 1].tick_params(axis='y', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
plt.show()
print("Comparison plots saved!")

# %% [markdown]
# ## 9. Select & Export Best Model

# %%
# Pick best
if cnn_acc >= dnn_acc:
    best_model, best_type, best_acc = cnn_model, "CNN", cnn_acc
else:
    best_model, best_type, best_acc = dnn_model, "DNN", dnn_acc
print(f"✅ Best model: {best_type} (Test Accuracy: {best_acc:.2f}%)")

# %%
# Save state dict
torch.save({
    "model_type": best_type,
    "state_dict": best_model.state_dict(),
    "accuracy": best_acc,
    "class_names": class_names,
    "input_shape": [1, 28, 28],
    "num_classes": 10,
}, os.path.join(MODEL_DIR, "best_model_state.pt"))
print("State dict saved ✓")

# %%
# Save TorchScript
best_model.eval()
example = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    traced = torch.jit.trace(best_model.cpu(), example)
traced.save(os.path.join(MODEL_DIR, "best_model.pt"))
print("TorchScript model saved ✓")

# %%
# Save preprocessor
preprocessor.save(os.path.join(MODEL_DIR, "preprocessor.joblib"))
print("Preprocessor saved ✓")

# %%
# Save metadata
metadata = {
    "model_type": best_type,
    "framework": "PyTorch",
    "accuracy": best_acc,
    "class_names": class_names,
    "input_shape": [1, 28, 28],
    "architecture": {
        "DNN": {"parameters": dnn_params, "test_accuracy": dnn_acc},
        "CNN": {"parameters": cnn_params, "test_accuracy": cnn_acc},
    },
    "comparison": {"dnn_acc": dnn_acc, "cnn_acc": cnn_acc, "best": best_type},
    "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
                 "optimizer": "Adam", "date": datetime.now().isoformat()},
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved ✓")

# %%
print(f"\n{'='*60}")
print("🏁 Training Complete! Files ready for deployment.")
print(f"{'='*60}")
print(f"  models/best_model.pt       — TorchScript model")
print(f"  models/best_model_state.pt — State dict checkpoint")
print(f"  models/preprocessor.joblib — Preprocessing pipeline")
print(f"  models/metadata.json       — Model metadata")
