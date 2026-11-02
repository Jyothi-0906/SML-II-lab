"""Q11: L1 (Lasso), L2 (Ridge) and Elastic Net regularization on a regression dataset."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# ---- 1. Data: 20 features, only 5 truly informative ----
X, y, true_coef = make_regression(n_samples=200, n_features=20, n_informative=5,
                                  noise=15, coef=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# ---- 2. Models ----
models = {
    "OLS (no reg.)":         LinearRegression(),
    "Ridge (L2)":            Ridge(alpha=10.0),
    "Lasso (L1)":            Lasso(alpha=1.0),
    "Elastic Net (L1+L2)":   ElasticNet(alpha=0.3, l1_ratio=0.5),
}

print(f"{'Model':<22}{'Train R2':>10}{'Test R2':>10}{'Test MSE':>12}{'Zero coefs':>12}")
coefs = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    coefs[name] = m.coef_
    print(f"{name:<22}{r2_score(y_train, m.predict(X_train)):>10.3f}"
          f"{r2_score(y_test, m.predict(X_test)):>10.3f}"
          f"{mean_squared_error(y_test, m.predict(X_test)):>12.1f}"
          f"{int(np.sum(np.abs(m.coef_) < 1e-6)):>12d}")

# ---- 3. Plot A: coefficient comparison ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
idx = np.arange(X.shape[1]); w = 0.2
for i, (name, c) in enumerate(coefs.items()):
    axes[0].bar(idx + i * w, c, w, label=name)
axes[0].plot(idx + 1.5 * w, true_coef, "k*", ms=9, label="True coef (unscaled)")
axes[0].set_title("Learned coefficients"); axes[0].set_xlabel("Feature index")
axes[0].set_ylabel("Coefficient"); axes[0].legend(fontsize=8)

# ---- 4. Plot B: regularization paths (coef vs alpha) ----
alphas = np.logspace(-2, 2.5, 60)
for ax, (title, cls, kw) in zip(axes[1:], [("Lasso (L1) path", Lasso, {}),
                                           ("Ridge (L2) path", Ridge, {})]):
    path = np.array([cls(alpha=a, **kw).fit(X_train, y_train).coef_ for a in alphas])
    ax.plot(alphas, path)
    ax.set_xscale("log"); ax.set_title(title)
    ax.set_xlabel("alpha (regularization strength)"); ax.set_ylabel("Coefficient")
plt.tight_layout(); plt.show()

# ---- 5. Plot C: test error and sparsity vs alpha ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, cls, kw in [("Ridge", Ridge, {}), ("Lasso", Lasso, {}),
                      ("ElasticNet", ElasticNet, {"l1_ratio": 0.5})]:
    mse, nz = [], []
    for a in alphas:
        m = cls(alpha=a, **kw).fit(X_train, y_train)
        mse.append(mean_squared_error(y_test, m.predict(X_test)))
        nz.append(np.sum(np.abs(m.coef_) > 1e-6))
    axes[0].plot(alphas, mse, label=name); axes[1].plot(alphas, nz, label=name)
axes[0].set(xscale="log", xlabel="alpha", ylabel="Test MSE", title="Test error vs alpha")
axes[1].set(xscale="log", xlabel="alpha", ylabel="# non-zero coefficients",
            title="Sparsity vs alpha")
for ax in axes: ax.legend()
plt.tight_layout(); plt.show()
