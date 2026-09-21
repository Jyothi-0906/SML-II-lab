"""
Q7: Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC for
    Logistic Regression and SVM -- on YOUR dataset (BSDS500 boundary task).

Same binary classification target used throughout this project: predict
whether a pixel is on a human-annotated object boundary, from its
color/gradient/texture features.

BACKGROUND ON THE METRICS
---------------------------
Given confusion-matrix counts TP, TN, FP, FN:

  Accuracy  = (TP+TN) / (TP+TN+FP+FN)      -- overall fraction correct.
  Precision = TP / (TP+FP)                 -- of predicted boundaries, how
                                               many really are boundaries?
  Recall    = TP / (TP+FN)                 -- of real boundaries, how many
                                               did we catch?
  F1-score  = 2*(Precision*Recall)/(Precision+Recall)   -- harmonic mean.
  ROC-AUC   = area under the ROC curve (True Positive Rate vs False
              Positive Rate across ALL thresholds); 1.0 = perfect ranking,
              0.5 = random guessing.

MODEL BACKGROUND
------------------
- Logistic Regression: linear decision boundary, fit by maximizing
  likelihood via the sigmoid function (see Q4 for full derivation).
- SVM (Support Vector Machine): finds the hyperplane maximizing the margin
  between classes' closest points ("support vectors"). With an RBF kernel,
  it implicitly maps data into a higher-dimensional space (the "kernel
  trick") to capture non-linear boundaries between edge and non-edge
  pixels -- likely to help here since boundary strength is not a purely
  linear function of raw pixel color/gradient values.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]

df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
X, y = df[FEATURES].values, df["boundary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
}

results = []
plt.figure(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-score": f1, "ROC-AUC": auc})

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["non-boundary", "boundary"]))

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Logistic Regression vs SVM\n(YOUR BSDS500 boundary-detection task)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q7_roc_curve.png", dpi=150)
print("\nSaved ROC curve comparison to outputs/q7_roc_curve.png")

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n=== Summary Comparison Table ===")
print(results_df)

results_df.to_csv("outputs/q7_comparison_table.csv")
print("\nSaved comparison table to outputs/q7_comparison_table.csv")

results_df.plot(kind="bar", figsize=(9, 5), rot=0)
plt.title("Logistic Regression vs SVM: Metric Comparison\n(YOUR BSDS500 boundary-detection task)")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/q7_metric_bars.png", dpi=150)
print("Saved metric bar chart to outputs/q7_metric_bars.png")
