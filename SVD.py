import numpy as np

# ----------- MATRIX INPUT ------------

print("Enter number of rows:")
r = int(input())

print("Enter number of columns:")
c = int(input())

print("\nEnter matrix elements row wise:")

A = []

for i in range(r):
    row = list(map(float,input(f"Row {i+1}: ").split()))
    A.append(row)

A = np.array(A)

print("\nMatrix A =")
print(A)

# ----------- EIGEN VALUES + EIGEN VECTORS ------------

print("\n========== Eigen Values and Eigen Vectors ==========")

eigen_values , eigen_vectors = np.linalg.eig(A)

print("\nEigen Values :")
print(eigen_values)

print("\nEigen Vectors (Column Wise) :")
print(eigen_vectors)


# ----------- SVD ------------

print("\n========== Singular Value Decomposition ==========")

U , singular_values , VT = np.linalg.svd(A)

print("\nU Matrix :")
print(U)

# Sigma Matrix Construction

Sigma = np.zeros((r,c))

for i in range(min(r,c)):
    Sigma[i][i] = singular_values[i]

print("\nSigma Matrix :")
print(Sigma)

print("\nV^T Matrix :")
print(VT)


# ----------- Verification ------------

print("\n========== Verification (U Σ V^T) ==========")

reconstructed = U @ Sigma @ VT

print("\nU Σ V^T =")
print(reconstructed)

print("\nOriginal Matrix =")
print(A)

print("\nDifference (Should be near zero):")
print(A - reconstructed)