import numpy as np

# -------------------------------------------------
# Matrix A
# -------------------------------------------------

A = np.array([
    [2, 3, 5],
    [4, 2, 4]
], dtype=float)

print("="*60)
print("Original Matrix")
print("="*60)
print(A)

# -------------------------------------------------
# Step 1
# Compute A^T A
# -------------------------------------------------

ATA = A.T @ A

print("\n" + "="*60)
print("A^T * A")
print("="*60)
print(ATA)

# -------------------------------------------------
# Step 2
# Eigenvalues and Eigenvectors
# -------------------------------------------------

eigenValues, eigenVectors = np.linalg.eigh(ATA)

# Sort in descending order
idx = np.argsort(eigenValues)[::-1]

eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:, idx]

print("\n" + "="*60)
print("Eigenvalues")
print("="*60)
print(eigenValues)

print("\n" + "="*60)
print("Eigenvectors")
print("="*60)
print(eigenVectors)

# -------------------------------------------------
# Step 3
# First Singular Value
# -------------------------------------------------

sigma1 = np.sqrt(eigenValues[0])

print("\n" + "="*60)
print("First Singular Value")
print("="*60)
print(sigma1)

# -------------------------------------------------
# Step 4
# First Right Singular Vector
# -------------------------------------------------

v1 = eigenVectors[:, 0]

print("\n" + "="*60)
print("First Right Singular Vector (v1)")
print("="*60)
print(v1)

# -------------------------------------------------
# Step 5
# First Left Singular Vector
# u = Av / sigma
# -------------------------------------------------

u1 = (A @ v1) / sigma1

print("\n" + "="*60)
print("First Left Singular Vector (u1)")
print("="*60)
print(u1)