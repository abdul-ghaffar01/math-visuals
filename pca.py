import numpy as np

# Dataset
X = np.array([
    [150, 50],
    [155, 53],
    [160, 58],
    [165, 60],
    [170, 65],
    [175, 70]
])

# Step 1: Center the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 2: Covariance matrix
cov_matrix = np.cov(X_centered.T)

# Step 3: Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Projection onto PC1
PC1 = eigenvectors[:, 0]
X_reduced = X_centered @ PC1

print("Centered Data:\n", X_centered)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nProjected Data:\n", X_reduced)