"""
Q2: Decision Tree classifier (scikit-learn) + tree visualization,
    trained on YOUR dataset (BSDS500 boundary-pixel classification).

Same dataset as Q1: pixel features (R,G,B,gray,grad_mag,local_std,laplacian)
-> label = is this pixel on a human-annotated object boundary?

ALGORITHM BACKGROUND
---------------------
A Decision Tree splits the feature space recursively into regions using
simple "if feature <= threshold" rules at each internal node, until leaves
are pure (single class) or a stopping condition is met.

How a split is chosen at each node:
  - For each feature and candidate threshold, measure how "impure" the
    resulting two child nodes would be.
  - Gini impurity:   G = 1 - sum(p_i^2)     (p_i = fraction of class i)
  - Entropy:         H = -sum(p_i * log2(p_i))
  - Pick the (feature, threshold) that gives the largest impurity decrease.
  - Repeat recursively for each child -> a tree grows.

Given our features, we'd expect grad_mag (Sobel gradient magnitude) and
laplacian to be the most useful splitters, since edges are precisely where
intensity changes sharply -- that's exactly what we'll check with feature
importances below.

Overfitting control: max_depth, min_samples_split, min_samples_leaf, or
post-pruning via ccp_alpha.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

FEATURES = ["R", "G", "B", "gray", "grad_mag", "local_std", "laplacian"]
CLASS_NAMES = ["non-boundary", "boundary"]

df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
X, y = df[FEATURES].values, df["boundary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc:.4f}\n")
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("\nClassification Report:\n",
      classification_report(y_test, preds, target_names=CLASS_NAMES))

print("\nFeature importances (which pixel features matter most for detecting edges):")
for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:<12s}: {imp:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=FEATURES, class_names=CLASS_NAMES,
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree: predicting boundary pixels in YOUR BSDS500 images (max_depth=4)")
plt.tight_layout()
plt.savefig("outputs/q2_decision_tree.png", dpi=150)
print("\nSaved tree visualization to outputs/q2_decision_tree.png")

depths = range(1, 15)
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train))
    test_accs.append(m.score(X_test, y_test))

plt.figure(figsize=(7, 5))
plt.plot(list(depths), train_accs, marker="o", label="Train accuracy")
plt.plot(list(depths), test_accs, marker="s", label="Test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting vs max_depth (BSDS500 boundary task)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/q2_depth_vs_accuracy.png", dpi=150)
print("Saved depth-vs-accuracy plot to outputs/q2_depth_vs_accuracy.png")
