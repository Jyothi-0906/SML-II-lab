"""
Q3: Naive Bayes classifier to detect Spam vs Ham
    -- adapted to run on YOUR dataset (BSDS500).

IMPORTANT NOTE ON THE DATA
----------------------------
"Spam vs Ham" is inherently a TEXT classification problem (it needs actual
email/SMS messages, which your uploaded dataset does not contain -- it's a
collection of natural images + boundary annotations, not text). There is no
way to honestly extract spam/ham labels from an image dataset.

What CAN be done, and what this script does, is keep the exercise 100%
faithful to your data: we solve the same *kind* of problem -- a BINARY
classification with Naive Bayes -- on the real binary target your dataset
actually provides: "is this pixel a boundary (edge) pixel, or a non-boundary
(flat/background) pixel?" This is exactly analogous in structure to
spam(1)/ham(0): two classes, and we use Bayes' theorem with an independence
assumption over features to decide between them.

ALGORITHM BACKGROUND
---------------------
Naive Bayes uses Bayes' Theorem:

        P(class | features) = P(features | class) * P(class) / P(features)

and "naively" assumes the features are conditionally independent given the
class:

        P(f1,...,fn | class) ~= P(f1|class) * P(f2|class) * ... * P(fn|class)

We pick the class maximizing P(class) * product(P(feature|class)).

For TEXT (spam/ham), features are word counts, so MultinomialNB is used,
modelling P(word|class) from word frequencies with Laplace smoothing.

For our CONTINUOUS pixel features (color, gradient magnitude, texture,
Laplacian), the appropriate variant is GAUSSIAN Naive Bayes, which assumes
each feature is normally distributed within each class:

        P(feature | class) = (1 / sqrt(2*pi*sigma_c^2)) *
                              exp(-(x - mu_c)^2 / (2*sigma_c^2))

where mu_c, sigma_c are the mean/std of that feature estimated from the
training examples of class c. This is the same "Naive Bayes" algorithm
family, just the right likelihood model for continuous data instead of
word counts.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]

df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
print(f"Loaded {len(df)} pixel samples from YOUR BSDS500 images.")
print(df["boundary"].value_counts(), "\n")

# quick EDA: how do feature distributions differ between the two classes?
print("Mean feature values by class (boundary=1 vs non-boundary=0):")
print(df.groupby("boundary")[FEATURES].mean(), "\n")

X = df[FEATURES].values
y = df["boundary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# GaussianNB doesn't strictly require scaling (it estimates its own mean/std
# per feature per class), but scaling still helps numerical stability.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

nb = GaussianNB()
nb.fit(X_train_s, y_train)
preds = nb.predict(X_test_s)

print("=== Gaussian Naive Bayes Results (boundary vs non-boundary pixels) ===")
print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds):.4f}")
print(f"Recall   : {recall_score(y_test, preds):.4f}")
print(f"F1-score : {f1_score(y_test, preds):.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted, [non-boundary, boundary]):")
print(confusion_matrix(y_test, preds))
print("\nClassification Report:\n",
      classification_report(y_test, preds, target_names=["non-boundary", "boundary"]))

# Inspect the learned per-class Gaussian parameters for each feature --
# this is literally what the model "learned" (analogous to word probabilities
# in text Naive Bayes).
print("Learned per-class mean for each feature (theta_):")
print(pd.DataFrame(nb.theta_, columns=FEATURES, index=["non-boundary", "boundary"]).round(3))
print("\nLearned per-class variance for each feature (var_):")
print(pd.DataFrame(nb.var_, columns=FEATURES, index=["non-boundary", "boundary"]).round(3))

# Try predicting on a couple of manually-picked pixels (sanity check)
sample_boundary_pixel = df[df.boundary == 1][FEATURES].iloc[0].values.reshape(1, -1)
sample_flat_pixel = df[df.boundary == 0][FEATURES].iloc[0].values.reshape(1, -1)
for label, pixel in [("known boundary pixel", sample_boundary_pixel),
                      ("known non-boundary pixel", sample_flat_pixel)]:
    pred = nb.predict(scaler.transform(pixel))[0]
    print(f"\n{label} -> model predicts: {'boundary' if pred == 1 else 'non-boundary'}")
