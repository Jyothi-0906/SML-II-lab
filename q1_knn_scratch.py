"""
Q1: k-Nearest Neighbours (k-NN) Classifier from scratch,
    tested on YOUR dataset (BSDS500 boundary-pixel classification).

DATASET USED (built by build_dataset.py from your images + .mat files)
-------------------------------------------------------------------------
Each row = one pixel sampled from your BSDS500 images.
Features = R, G, B, gray intensity, Sobel gradient magnitude, local
           neighborhood std (texture), Laplacian response.
Label    = 'boundary' (1 if >=3 of 5 human annotators marked this pixel as
           an object boundary/edge, 0 otherwise).

So the task k-NN solves here is: "given a pixel's local color/texture/edge
signature, is it on an object boundary or not?" -- this is literally the
classic problem BSDS500 was built for.

ALGORITHM BACKGROUND
---------------------
k-NN is a "lazy", instance-based learner: it does not build an explicit
model during training, it just memorizes the training data. To classify a
new point:
  1. Compute the distance from it to every training point (Euclidean here).
  2. Take the k closest training points ("neighbours").
  3. Return the majority class among those k neighbours.

Key ideas:
  - k is a hyperparameter: small k -> low bias/high variance (overfits to
    noise); large k -> high bias/low variance (smoother boundary).
  - Feature scaling matters because distance is sensitive to magnitude
    (gradient magnitude can be 0-1000+, while gray is 0-255, etc.) so we
    standardize before computing distance.
  - Complexity: O(n*d) per query -- every prediction scans the whole
    training set, which is why it's "lazy"/expensive at inference time.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]


class KNNFromScratch:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def _predict_one(self, x):
        distances = self._euclidean(self.X_train, x)
        k_idx = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_idx]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == np.asarray(y))


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
    print(f"Loaded {len(df)} pixel samples from YOUR BSDS500 images.")
    print(df["boundary"].value_counts(), "\n")

    X = df[FEATURES].values
    y = df["boundary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("k  | Test Accuracy")
    print("-------------------")
    best_k, best_acc = None, -1
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        model = KNNFromScratch(k=k).fit(X_train_s, y_train)
        acc = model.score(X_test_s, y_test)
        print(f"{k:<3}| {acc:.4f}")
        if acc > best_acc:
            best_acc, best_k = acc, k

    print(f"\nBest k = {best_k} with test accuracy = {best_acc:.4f}")

    final_model = KNNFromScratch(k=best_k).fit(X_train_s, y_train)
    preds = final_model.predict(X_test_s)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nConfusion Matrix (rows=actual, cols=predicted, [non-boundary, boundary]):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:\n",
          classification_report(y_test, preds, target_names=["non-boundary", "boundary"]))

    from sklearn.neighbors import KNeighborsClassifier
    sk_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_s, y_train)
    sk_acc = sk_model.score(X_test_s, y_test)
    print(f"\nSanity check -- sklearn KNeighborsClassifier accuracy: {sk_acc:.4f}")
    print("(Should match our from-scratch implementation closely.)")
