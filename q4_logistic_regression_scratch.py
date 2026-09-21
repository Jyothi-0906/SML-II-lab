"""
Q4: Logistic Regression implemented from scratch using Gradient Descent,
    trained on YOUR dataset (BSDS500 boundary-pixel classification).

Same binary target as Q1-Q3: is this pixel on a human-annotated object
boundary (1) or not (0), based on its color/gradient/texture features.

ALGORITHM BACKGROUND
---------------------
Logistic Regression is a linear model for BINARY classification:

        z = w.x + b
        p = sigmoid(z) = 1 / (1 + e^-z)         (squashes z into (0,1))

Predict class 1 if p >= 0.5, else class 0.

TRAINING: minimize the (binary) Log-Loss / Cross-Entropy cost, averaged
over m training examples:

        J(w,b) = -(1/m) * sum[ y*log(p) + (1-y)*log(1-p) ]

This cost is convex in (w,b), so Gradient Descent finds the global minimum.
The gradients work out to:

        dJ/dw = (1/m) * X^T . (p - y)
        dJ/db = (1/m) * sum(p - y)

GRADIENT DESCENT update rule (repeated for many iterations):
        w := w - learning_rate * dJ/dw
        b := b - learning_rate * dJ/db

Feature scaling (standardization) is important here too -- it makes the
loss surface more spherical so gradient descent converges quickly and
without oscillation.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iters=3000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.losses = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        y = np.asarray(y)

        for i in range(self.n_iters):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (p - y))
            db = (1 / n_samples) * np.sum(p - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            eps = 1e-15
            p_clipped = np.clip(p, eps, 1 - eps)
            loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            self.losses.append(loss)
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
    X, y = df[FEATURES].values, df["boundary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegressionScratch(learning_rate=0.5, n_iters=3000)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy (from-scratch Logistic Regression): {acc:.4f}\n")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification Report:\n",
          classification_report(y_test, preds, target_names=["non-boundary", "boundary"]))

    print("\nLearned weights (which features push the model toward 'boundary'):")
    for name, w in sorted(zip(FEATURES, model.w), key=lambda t: -abs(t[1])):
        print(f"  {name:<12s}: {w:+.4f}")
    print(f"  bias        : {model.b:+.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(model.losses)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Logistic Regression (from scratch) on BSDS500 boundary task:\nLoss vs Gradient Descent Iterations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/q4_gd_convergence.png", dpi=150)
    print("\nSaved convergence plot to outputs/q4_gd_convergence.png")

    from sklearn.linear_model import LogisticRegression
    sk_model = LogisticRegression(max_iter=5000).fit(X_train_s, y_train)
    sk_acc = sk_model.score(X_test_s, y_test)
    print(f"\nSanity check -- sklearn LogisticRegression accuracy: {sk_acc:.4f}")
