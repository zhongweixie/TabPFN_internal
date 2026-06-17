#!/usr/bin/env python3
"""
Simple test to verify TabPFN-3 can load local models
"""
import os
os.environ["TABPFN_NO_BROWSER"] = "1"  # Disable browser authentication

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import time

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Try with local model
model_path = "/home/zxiebk/workspace/model/tabpfn_3/tabpfn-v3-classifier-v3_default.ckpt"
print(f"Using model: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

clf = TabPFNClassifier(
    device="cuda",
    model_path=model_path
)

start = time.time()
clf.fit(X_train, y_train)
fit_time = time.time() - start

start = time.time()
y_pred = clf.predict(X_test)
pred_time = time.time() - start

acc = accuracy_score(y_test, y_pred)

print(f"\n✓ Accuracy: {acc:.4f}")
print(f"✓ Fit time: {fit_time:.3f}s")
print(f"✓ Predict time: {pred_time:.3f}s")
