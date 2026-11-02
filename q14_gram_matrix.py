"""Q14: Compute and visualise the Gram matrix of a dataset using different kernels."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import (linear_kernel, polynomial_kernel, rbf_kernel,
                                      sigmoid_kernel, laplacian_kernel)

# ---- Data: sorted by class so block structure is visible in the heatmaps ----
X, y = make_moons(n_samples=80, noise=0.08, random_state=1)
order = np.argsort(y)
X, y = X[order], y[order]

# ---- Gram matrices K[i, j] = k(x_i, x_j) ----
kernels = {
    "Linear: x·y":                    linear_kernel(X),
    "Polynomial (deg=3)":             polynomial_kernel(X, degree=3, gamma=1, coef0=1),
    "RBF (gamma=1)":                  rbf_kernel(X, gamma=1.0),
    "RBF (gamma=10)":                 rbf_kernel(X, gamma=10.0),
    "Laplacian (gamma=1)":            laplacian_kernel(X, gamma=1.0),
    "Sigmoid (tanh)":                 sigmoid_kernel(X, gamma=0.5, coef0=0),
}

# ---- Manual RBF implementation as a sanity check ----
sq = np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2 * X @ X.T
assert np.allclose(np.exp(-1.0 * sq), kernels["RBF (gamma=1)"]), "RBF mismatch"

# ---- Properties: symmetry and positive semi-definiteness ----
print(f"{'Kernel':<24}{'Symmetric':>10}{'Min eigenvalue':>18}{'Valid (PSD)':>14}")
for name, K in kernels.items():
    min_eig = np.linalg.eigvalsh(K).min()
    print(f"{name:<24}{str(np.allclose(K, K.T)):>10}{min_eig:>18.4f}{str(min_eig > -1e-8):>14}")

# ---- Visualise ----
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.ravel()
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=25)
axes[0].set_title("Dataset (two moons)")
n0 = np.sum(y == 0)
for ax, (name, K) in zip(axes[1:], kernels.items()):
    im = ax.imshow(K, cmap="viridis")
    ax.axhline(n0 - 0.5, color="w", lw=0.8); ax.axvline(n0 - 0.5, color="w", lw=0.8)
    ax.set_title(f"Gram matrix: {name}")
    ax.set_xlabel("sample j"); ax.set_ylabel("sample i")
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[-1].axis("off")
plt.suptitle("Gram matrices (white lines separate class 0 / class 1; samples sorted by class)")
plt.tight_layout(); plt.show()
