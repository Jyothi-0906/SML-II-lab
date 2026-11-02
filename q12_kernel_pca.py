"""Q12: Kernel PCA on a nonlinear dataset and visualisation of class separation."""
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA, KernelPCA

datasets = {
    "Concentric circles": make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0),
    "Two moons":          make_moons(n_samples=400, noise=0.07, random_state=0),
}

fig, axes = plt.subplots(len(datasets), 4, figsize=(20, 9))
for row, (name, (X, y)) in enumerate(datasets.items()):
    projections = {
        "Original data": X,
        "Linear PCA": PCA(n_components=2).fit_transform(X),
        "Kernel PCA (RBF, gamma=10)": KernelPCA(n_components=2, kernel="rbf", gamma=10).fit_transform(X),
        "Kernel PCA (poly, deg=3)": KernelPCA(n_components=2, kernel="poly", degree=3,
                                              gamma=1, coef0=1).fit_transform(X),
    }
    for col, (title, Z) in enumerate(projections.items()):
        ax = axes[row, col]
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c="tab:red", s=15, alpha=0.7, label="Class 0")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c="tab:blue", s=15, alpha=0.7, label="Class 1")
        ax.set_title(f"{name}\n{title}" if col == 0 else title, fontsize=10)
        ax.set_xlabel("Component 1" if col else "x1")
        ax.set_ylabel("Component 2" if col else "x2")
        if row == 0 and col == 0: ax.legend()
plt.suptitle("Linear PCA vs Kernel PCA on nonlinear data", fontsize=14)
plt.tight_layout(); plt.show()

# ---- Quantify separation: linear classifier on the 1st component only ----
from sklearn.linear_model import LogisticRegression
X, y = datasets["Concentric circles"]
for label, Z in [("Linear PCA", PCA(n_components=1).fit_transform(X)),
                 ("RBF Kernel PCA", KernelPCA(n_components=1, kernel="rbf", gamma=10).fit_transform(X))]:
    acc = LogisticRegression().fit(Z, y).score(Z, y)
    print(f"{label:<16} -> accuracy using only 1 component (circles): {acc:.3f}")
