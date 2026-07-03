"""
Test script for FashionMNIST API.
Run: python test_api.py
"""

import requests
import numpy as np
import json
import sys

# Configuration
BASE_URL = "http://localhost:8080"
API_KEY = "fashion-mnist-api-key-2026"  # Default; change if set differently

# Test 1: Root endpoint
print("=" * 60)
print("Test 1: GET /")
response = requests.get(f"{BASE_URL}/")
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# Test 2: Health
print("\n" + "=" * 60)
print("Test 2: GET /health")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# Test 3: Predict with random image
print("\n" + "=" * 60)
print("Test 3: POST /predict (with random image)")
random_image = np.random.randint(0, 256, (784,)).tolist()
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}
payload = {"image": random_image}
response = requests.post(f"{BASE_URL}/predict", json=payload, headers=headers)
print(f"Status: {response.status_code}")
if response.ok:
    result = response.json()
    print(f"Predicted: {result['class_name']} (class {result['predicted_class']})")
    print(f"Confidence: {result['confidence']}")
    print(f"All probabilities: {result['all_probabilities']}")
else:
    print(f"Error: {response.text}")

# Test 4: Missing API key
print("\n" + "=" * 60)
print("Test 4: POST /predict (no API key — should fail)")
response = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"Status: {response.status_code}")
print(f"Detail: {response.json().get('detail', 'N/A')}")

# Test 5: Invalid API key
print("\n" + "=" * 60)
print("Test 5: POST /predict (wrong API key — should fail)")
response = requests.post(f"{BASE_URL}/predict", json=payload, headers={"X-API-Key": "wrong-key"})
print(f"Status: {response.status_code}")
print(f"Detail: {response.json().get('detail', 'N/A')}")

# Test 6: Predict with a real sample
print("\n" + "=" * 60)
print("Test 6: POST /predict (with test FashionMNIST sample)")
from torchvision import datasets, transforms
import torch

test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True,
    transform=transforms.ToTensor()
)
sample_img, sample_label = test_dataset[888]
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Convert to flat list of 0-255 values
img_flat = (sample_img.numpy() * 255).astype(int).flatten().tolist()
payload = {"image": img_flat}
response = requests.post(f"{BASE_URL}/predict", json=payload, headers=headers)
print(f"Actual label: {class_names[sample_label]}")
if response.ok:
    result = response.json()
    print(f"Predicted:   {result['class_name']}")
    print(f"Confidence:  {result['confidence']}")
    print(f"Correct?     {'✅ YES' if result['predicted_class'] == sample_label else '❌ NO'}")
else:
    print(f"Error: {response.text}")

print("\n" + "=" * 60)
print("All tests complete!")
