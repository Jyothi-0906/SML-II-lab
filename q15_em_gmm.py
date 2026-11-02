"""EM algorithm for fitting a Gaussian Mixture Model (GMM), implemented from scratch."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs

np.random.seed(42)

# ---- 1. Data: 3 overlapping 2-D clusters ----
X, true_labels = make_blobs(n_samples=400, centers=3, cluster_std=[1.0, 1.5, 0.8],
                            random_state=7)
n, d = X.shape
K = 3


# ---- 2. Gaussian density N(x | mean, cov) for every row of X ----
def gaussian_pdf(X, mean, cov):
    diff = X - mean
    inv = np.linalg.inv(cov)
    norm = 1.0 / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(cov))
    return norm * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))


# ---- 3. EM algorithm ----
def fit_gmm(X, K, max_iter=100, tol=1e-6):
    n, d = X.shape
    # Initialise: random data points as means, equal weights, shared covariance
    means = X[np.random.choice(n, K, replace=False)]
    covs = np.array([np.cov(X.T) for _ in range(K)])
    weights = np.ones(K) / K
    log_likelihoods = []

    for it in range(max_iter):
        # E-step: responsibility of each component for each point
        weighted = np.column_stack([weights[k] * gaussian_pdf(X, means[k], covs[k])
                                    for k in range(K)])
        total = weighted.sum(axis=1, keepdims=True)
        resp = weighted / total                      # shape (n, K)

        # Log-likelihood of the data under the current parameters
        log_likelihoods.append(np.sum(np.log(total)))

        # M-step: re-estimate weights, means and covariances
        Nk = resp.sum(axis=0)                        # effective points per component
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - means[k]
            covs[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]
            covs[k] += 1e-6 * np.eye(d)              # keeps covariance invertible

        # Stop when the log-likelihood no longer improves
        if it > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return weights, means, covs, resp, log_likelihoods


weights, means, covs, resp, log_likelihoods = fit_gmm(X, K)
labels = resp.argmax(axis=1)                         # hard assignment

# ---- 4. Results ----
print(f"Converged after {len(log_likelihoods)} iterations")
print(f"Final log-likelihood: {log_likelihoods[-1]:.2f}")
for k in range(K):
    print(f"Component {k}: weight = {weights[k]:.3f}, mean = {np.round(means[k], 2)}")


# ---- 5. Plots ----
def draw_ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    width, height = 2 * 2 * np.sqrt(vals[::-1])      # 2 standard deviations
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, color=color, lw=2))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ["tab:red", "tab:blue", "tab:green"]

axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap="coolwarm", s=15)
axes[0].set_title("True clusters")

axes[1].scatter(X[:, 0], X[:, 1], c=[colors[l] for l in labels], s=15, alpha=0.7)
for k in range(K):
    axes[1].plot(*means[k], "k*", ms=14)
    draw_ellipse(axes[1], means[k], covs[k], colors[k])
axes[1].set_title("GMM fitted by EM (ellipses = 2 std. dev.)")

axes[2].plot(range(1, len(log_likelihoods) + 1), log_likelihoods, "o-")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Log-likelihood")
axes[2].set_title("EM convergence")

for ax in axes[:2]:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
plt.tight_layout()
plt.show()
