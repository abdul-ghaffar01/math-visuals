#!/usr/bin/env python3
# =============================================================================
# PCA VISUALIZER
# -----------------------------------------------------------------------------
# An educational, single-file Python program that teaches Principal Component
# Analysis (PCA) from first principles, built entirely from scratch on top of
# numpy's linear algebra routines -- no scikit-learn, no scipy.
#
# Run with:
#     python pca_visualizer.py
#
# Allowed libraries: numpy, matplotlib, and the Python standard library
# (os, sys, time, textwrap) used purely for terminal presentation.
# Deliberately NOT used: sklearn, scipy, pandas, seaborn, OpenCV.
#
# This program is a companion to the SVD Image Compression Visualizer -- PCA
# and SVD are two views of the exact same underlying mathematics, and
# Chapter 6 below makes that connection explicit.
# =============================================================================

import os
import sys
import time
import textwrap
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


# =============================================================================
# SECTION 0 : TERMINAL PRESENTATION HELPERS
# -----------------------------------------------------------------------------
# Purely cosmetic: section dividers, colored text, progress bars, and pauses.
# No mathematics happens in this section.
# =============================================================================

class Colors:
    """ANSI escape codes for colored terminal text."""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def supports_color() -> bool:
    """Return True if the current terminal likely supports ANSI colors."""
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    is_windows = sys.platform.startswith("win")
    return is_a_tty and not is_windows or (is_windows and "ANSICON" in os.environ)


USE_COLOR = supports_color()


def colored(text: str, color: str) -> str:
    """Wrap `text` in an ANSI color code if the terminal supports it."""
    if not USE_COLOR:
        return text
    return f"{color}{text}{Colors.END}"


def print_header(title: str) -> None:
    """Print a large section divider, used for each CHAPTER of the program."""
    line = "=" * 79
    print("\n" + colored(line, Colors.CYAN))
    print(colored(title.center(79), Colors.BOLD + Colors.CYAN))
    print(colored(line, Colors.CYAN))


def print_subheader(title: str) -> None:
    """Print a smaller divider used for sub-sections inside a chapter."""
    line = "-" * 60
    print("\n" + colored(title, Colors.YELLOW))
    print(colored(line, Colors.YELLOW))


def print_explanation(text: str) -> None:
    """Print a wrapped block of explanatory / educational text."""
    wrapped = textwrap.fill(text, width=78)
    print(colored(wrapped, Colors.GREEN))


def pause() -> None:
    """Pause execution so the reader can study the math before moving on."""
    try:
        input(colored("\nPress ENTER to continue...", Colors.BOLD))
    except EOFError:
        pass


def ascii_progress_bar(percentage: float, width: int = 40) -> str:
    """Build a simple ASCII progress bar string for a given percentage."""
    percentage = max(0.0, min(100.0, percentage))
    filled = int(round(width * percentage / 100.0))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {percentage:6.2f}%"


# =============================================================================
# SECTION 1 : SYNTHETIC DATA GENERATION
# -----------------------------------------------------------------------------
# CHAPTER 1 needs data to work with. Instead of loading a file, we generate
# two synthetic datasets using only numpy:
#   - A 2D dataset with two correlated clusters (easy to visualize PCA axes)
#   - A 3D dataset that mostly lives near a 2D plane, plus a little noise in
#     the third dimension (a natural example of "intrinsic low dimensionality")
# =============================================================================

def generate_2d_correlated_clusters(n_per_cluster: int = 150, seed: int = 42) -> np.ndarray:
    """Generate a 2D dataset made of two correlated Gaussian clusters.

    Each cluster is drawn from an independent 2D Gaussian, then stretched
    and rotated so that feature 1 and feature 2 become correlated -- this
    correlation is exactly what PCA will discover and exploit.
    """
    rng = np.random.default_rng(seed)

    # A rotation + stretch matrix creates correlation between the two axes.
    stretch = np.array([[3.0, 0.0], [0.0, 0.6]])
    theta = np.radians(35)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    transform = rotation @ stretch

    cluster_a = rng.standard_normal((n_per_cluster, 2)) @ transform.T + np.array([2.0, 1.0])
    cluster_b = rng.standard_normal((n_per_cluster, 2)) @ transform.T + np.array([-3.0, -2.0])

    data = np.vstack([cluster_a, cluster_b])
    rng.shuffle(data)
    return data


def generate_3d_near_planar_data(n_points: int = 300, seed: int = 7) -> np.ndarray:
    """Generate a 3D dataset whose points lie close to a tilted 2D plane.

    We first sample points on a 2D plane (using two independent directions),
    embed that plane into 3D space, and then add a small amount of noise in
    every direction. Because the noise is small compared to the in-plane
    spread, PCA will find that 2 components already explain almost all of
    the variance -- a clean, visual demonstration of "intrinsic
    dimensionality" being lower than the number of raw features.
    """
    rng = np.random.default_rng(seed)

    # Two independent in-plane directions (not necessarily orthogonal --
    # PCA will find the orthogonal directions that best explain the spread).
    direction_1 = np.array([1.0, 0.5, 0.3])
    direction_2 = np.array([-0.4, 1.0, 0.2])

    coeff_1 = rng.standard_normal(n_points) * 4.0
    coeff_2 = rng.standard_normal(n_points) * 2.0

    plane_points = (coeff_1[:, None] * direction_1) + (coeff_2[:, None] * direction_2)

    # Small isotropic noise pushes points slightly off the plane.
    noise = rng.standard_normal((n_points, 3)) * 0.3

    data = plane_points + noise
    return data


# =============================================================================
# CHAPTER 1 : WHAT IS DATA, MATHEMATICALLY?
# =============================================================================

def chapter1_what_is_data() -> Tuple[np.ndarray, np.ndarray]:
    """Explain that a dataset is simply a matrix, and generate our examples."""
    print_header("CHAPTER 1 : WHAT IS DATA, MATHEMATICALLY?")

    print_explanation(
        "Just like a grayscale image is secretly a matrix of pixel "
        "brightnesses, a dataset is secretly a matrix of measurements. "
        "Every row is one 'sample' (one observation, one data point), and "
        "every column is one 'feature' (one measured quantity). A dataset "
        "of n samples and d features is therefore an (n x d) matrix X. "
        "PCA only ever looks at this matrix -- it has no notion of what "
        "the features 'mean' in the real world, only how they vary "
        "together numerically."
    )

    data_2d = generate_2d_correlated_clusters()
    data_3d = generate_3d_near_planar_data()

    print_subheader("Dataset A: 2D Correlated Clusters")
    n2, d2 = data_2d.shape
    print(f"  Samples (rows)   : {n2}")
    print(f"  Features (cols)  : {d2}")
    print(f"  Total numbers    : {n2 * d2}")

    print_subheader("Dataset B: 3D Near-Planar Data")
    n3, d3 = data_3d.shape
    print(f"  Samples (rows)   : {n3}")
    print(f"  Features (cols)  : {d3}")
    print(f"  Total numbers    : {n3 * d3}")

    print_explanation(
        "Dataset A has two clusters whose features are correlated -- when "
        "feature 1 goes up, feature 2 tends to go up too, along a specific "
        "diagonal direction. Dataset B lives in 3 dimensions but is "
        "secretly 'almost 2-dimensional': its points scatter mostly within "
        "a tilted flat plane. PCA will discover both of these hidden "
        "structures automatically."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(data_2d[:, 0], data_2d[:, 1], s=15, alpha=0.6, color="tab:blue")
    axes[0].set_title("Dataset A: 2D Correlated Clusters")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)
    axes[0].axis("equal")

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax3d.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], s=12, alpha=0.6, color="tab:orange")
    ax3d.set_title("Dataset B: 3D Near-Planar Data")
    ax3d.set_xlabel("Feature 1")
    ax3d.set_ylabel("Feature 2")
    ax3d.set_zlabel("Feature 3")

    plt.tight_layout()
    plt.show()

    pause()
    return data_2d, data_3d


# =============================================================================
# CHAPTER 2 : THE DATA MATRIX
# =============================================================================

def chapter2_print_matrix(data: np.ndarray, name: str, crop_rows: int = 8) -> None:
    """Print the raw data matrix (or a small crop of it if it's large)."""
    print_header(f"CHAPTER 2 : THE DATA MATRIX ({name})")

    print_explanation(
        "Each entry X[i, j] is the value of feature j for sample i. Rows "
        "run down the samples; columns run across the features. There is "
        "nothing more mysterious in a dataset than this grid of numbers -- "
        "everything PCA does below is pure matrix arithmetic on this table."
    )

    n, d = data.shape
    rows_to_show = min(crop_rows, n)

    print_subheader(f"First {rows_to_show} rows of the ({n} x {d}) data matrix")
    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print(data[:rows_to_show])

    pause()


# =============================================================================
# CHAPTER 3 : MEAN CENTERING
# =============================================================================

def center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract the per-feature mean from every sample.

    PCA is defined around the *spread* of the data, not its absolute
    position in space. Centering shifts the data so its mean sits at the
    origin, which is a required first step before computing covariance or
    running SVD for PCA purposes.
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return X_centered, mean


def chapter3_mean_centering(data: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """CHAPTER 3: explain and perform mean centering."""
    print_header(f"CHAPTER 3 : MEAN CENTERING ({name})")

    print_explanation(
        "Before finding principal components, we must center the data: "
        "subtract the mean of each feature (each column) from every "
        "sample. This moves the 'center of mass' of the data cloud to the "
        "origin (0, 0, ...). PCA cares about how data varies AROUND its "
        "center, not where that center happens to sit -- centering removes "
        "the position information so only the spread and orientation "
        "remain."
    )

    centered, mean = center_data(data)

    print_subheader("Feature Means (before centering)")
    with np.printoptions(precision=4, suppress=True):
        print(f"  Mean vector: {mean}")

    if data.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].scatter(data[:, 0], data[:, 1], s=15, alpha=0.6, color="tab:blue")
        axes[0].scatter(*mean, color="red", marker="x", s=120, label="Mean")
        axes[0].set_title("Before Centering")
        axes[0].legend()
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(centered[:, 0], centered[:, 1], s=15, alpha=0.6, color="tab:green")
        axes[1].scatter(0, 0, color="red", marker="x", s=120, label="Origin")
        axes[1].set_title("After Centering")
        axes[1].legend()
        axes[1].axis("equal")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    pause()
    return centered, mean


# =============================================================================
# CHAPTER 4 : THE COVARIANCE MATRIX
# =============================================================================

def compute_covariance_matrix(X_centered: np.ndarray) -> np.ndarray:
    """Compute the (d x d) covariance matrix of centered data X.

    Formula: Cov = (X_centered^T @ X_centered) / (n - 1)

    The diagonal entries Cov[j, j] are the variance of feature j. The
    off-diagonal entries Cov[i, j] measure how features i and j vary
    together: positive means they tend to increase together, negative
    means one tends to increase as the other decreases, and near-zero
    means they are roughly independent.
    """
    n = X_centered.shape[0]
    covariance = (X_centered.T @ X_centered) / (n - 1)
    return covariance


def chapter4_covariance_matrix(X_centered: np.ndarray, name: str) -> np.ndarray:
    """CHAPTER 4: compute and explain the covariance matrix."""
    print_header(f"CHAPTER 4 : THE COVARIANCE MATRIX ({name})")

    print_explanation(
        "The covariance matrix summarizes how every pair of features "
        "varies together, in a single (d x d) table. It is computed as "
        "(X_centered^T @ X_centered) / (n - 1). The diagonal holds each "
        "feature's own variance (how spread out it is); the off-diagonal "
        "entries reveal correlation structure -- exactly the "
        "'diagonal stretching plus rotation' directions that PCA is about "
        "to uncover."
    )

    covariance = compute_covariance_matrix(X_centered)

    print_subheader(f"Covariance Matrix ({covariance.shape[0]} x {covariance.shape[1]})")
    with np.printoptions(precision=4, suppress=True):
        print(covariance)

    print_explanation(
        "Notice the off-diagonal values are not zero -- this confirms "
        "that our features are correlated. If the features were perfectly "
        "independent, the covariance matrix would be purely diagonal, and "
        "PCA would simply return the original axes unchanged."
    )

    pause()
    return covariance


# =============================================================================
# CHAPTER 5 : EIGEN DECOMPOSITION OF THE COVARIANCE MATRIX
# =============================================================================

def eigen_decomposition(covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of a symmetric covariance matrix.

    We use np.linalg.eigh (not the general np.linalg.eig) because a
    covariance matrix is always real and symmetric, and eigh is both
    faster and numerically more stable for symmetric matrices. eigh
    returns eigenvalues in ASCENDING order, so we reverse them (and their
    matching eigenvectors) to get the conventional descending order.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]  # descending order
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


def chapter5_eigen_decomposition(covariance: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """CHAPTER 5: run the eigen decomposition and explain what it means."""
    print_header(f"CHAPTER 5 : EIGEN DECOMPOSITION ({name})")

    print_explanation(
        "The principal components are the EIGENVECTORS of the covariance "
        "matrix, and the amount of variance each one explains is its "
        "EIGENVALUE. An eigenvector v of the covariance matrix satisfies "
        "Cov @ v = lambda * v -- meaning that when the covariance matrix "
        "'acts on' this special direction v, the result is just v "
        "stretched by a scalar lambda, with no rotation at all. These are "
        "precisely the directions along which the data spreads out "
        "independently of any other direction."
    )

    print_subheader("Calling np.linalg.eigh(covariance)")
    eigenvalues, eigenvectors = eigen_decomposition(covariance)

    print_subheader("Eigenvalues (variance explained by each component)")
    with np.printoptions(precision=4, suppress=True):
        print(f"  {eigenvalues}")

    print_subheader("Eigenvectors (principal component directions, as columns)")
    with np.printoptions(precision=4, suppress=True):
        print(eigenvectors)

    print_explanation(
        "Each column of the eigenvector matrix is one principal "
        "component -- a unit-length direction in feature space. The first "
        "column (largest eigenvalue) is 'Principal Component 1' (PC1): "
        "the single direction along which the data varies the most. The "
        "second column is PC2, the direction of second-most variance, "
        "constrained to be perpendicular (orthogonal) to PC1. And so on."
    )

    pause()
    return eigenvalues, eigenvectors


# =============================================================================
# CHAPTER 6 : THE CONNECTION BETWEEN PCA AND SVD
# =============================================================================

def chapter6_connection_to_svd(X_centered: np.ndarray, eigenvalues: np.ndarray,
                                 eigenvectors: np.ndarray, name: str) -> None:
    """CHAPTER 6: show that PCA and SVD compute the same thing two ways."""
    print_header(f"CHAPTER 6 : PCA IS SECRETLY SVD ({name})")

    print_explanation(
        "If you built the SVD Image Compression Visualizer, this chapter "
        "will feel very familiar: PCA and SVD are the same mathematics "
        "viewed from two different angles. Instead of eigen-decomposing "
        "the covariance matrix, we can run SVD directly on the CENTERED "
        "data matrix: X_centered = U @ diag(S) @ VT. It turns out that the "
        "rows of VT are exactly the principal component directions, and "
        "the singular values S relate to the eigenvalues by "
        "lambda_i = S_i^2 / (n - 1)."
    )

    n = X_centered.shape[0]
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues_from_svd = (S ** 2) / (n - 1)

    print_subheader("Eigenvalues from covariance matrix (Chapter 5)")
    with np.printoptions(precision=4, suppress=True):
        print(f"  {eigenvalues}")

    print_subheader("Eigenvalues recovered from SVD: (S^2) / (n - 1)")
    with np.printoptions(precision=4, suppress=True):
        print(f"  {eigenvalues_from_svd}")

    max_diff = float(np.max(np.abs(eigenvalues - eigenvalues_from_svd)))
    print_subheader("Agreement Check")
    print(f"  Maximum difference between the two eigenvalue lists: {max_diff:.10f}")
    print(colored("  (This should be extremely close to 0 -- they are the same numbers.)", Colors.CYAN))

    print_explanation(
        "This is why, in practice, most real PCA implementations "
        "(including the one inside scikit-learn) run SVD on the centered "
        "data rather than explicitly building the covariance matrix and "
        "eigen-decomposing it -- SVD is more numerically stable, "
        "especially when the number of features is large. Both routes "
        "arrive at the exact same principal component directions."
    )

    pause()


# =============================================================================
# CHAPTER 7 : SORTING & INTERPRETING PRINCIPAL COMPONENTS
# =============================================================================

def chapter7_interpret_components(eigenvalues: np.ndarray, name: str) -> None:
    """CHAPTER 7: print sorted eigenvalues and key statistics."""
    print_header(f"CHAPTER 7 : INTERPRETING THE PRINCIPAL COMPONENTS ({name})")

    print_explanation(
        "Eigenvalues are always sorted from largest to smallest so that "
        "'Principal Component 1' always refers to the most important "
        "direction. A large eigenvalue means that direction captures a "
        "lot of the dataset's spread; a small eigenvalue means that "
        "direction barely matters and could likely be dropped without "
        "losing much information."
    )

    print_subheader("Sorted Eigenvalues (Variance per Component)")
    for i, val in enumerate(eigenvalues, start=1):
        print(f"  PC{i}: {val:.4f}")

    print_subheader("Key Statistics")
    print(f"  Largest eigenvalue  (PC1) : {eigenvalues[0]:.4f}")
    print(f"  Smallest eigenvalue       : {eigenvalues[-1]:.4f}")
    print(f"  Number of components      : {len(eigenvalues)}")

    pause()


# =============================================================================
# CHAPTER 8 : EXPLAINED VARIANCE RATIO & SCREE PLOT
# =============================================================================

def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Fraction of total variance explained by each principal component."""
    total = np.sum(eigenvalues)
    return eigenvalues / total


def chapter8_explained_variance(eigenvalues: np.ndarray, name: str) -> np.ndarray:
    """CHAPTER 8: explained variance ratio, cumulative variance, scree plot."""
    print_header(f"CHAPTER 8 : EXPLAINED VARIANCE ({name})")

    print_explanation(
        "'Explained variance ratio' is the eigenvalue of a component "
        "divided by the sum of all eigenvalues -- it tells you what "
        "percentage of the dataset's total spread that one direction "
        "accounts for. 'Cumulative explained variance' adds these "
        "percentages up as you include more components, exactly like the "
        "cumulative energy curve from SVD image compression."
    )

    ratios = explained_variance_ratio(eigenvalues)
    cumulative = np.cumsum(ratios) * 100

    print_subheader("Per-Component and Cumulative Explained Variance")
    for i, (r, c) in enumerate(zip(ratios, cumulative), start=1):
        bar = ascii_progress_bar(c)
        print(f"  PC1-PC{i}: {bar}  (this component alone: {r * 100:5.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    component_indices = np.arange(1, len(eigenvalues) + 1)

    axes[0].bar(component_indices, ratios * 100, color="tab:blue")
    axes[0].set_title("Scree Plot: Variance Explained per Component")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_xticks(component_indices)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(component_indices, cumulative, marker="o", color="tab:green")
    axes[1].axhline(90, color="gray", linestyle="--", linewidth=1, label="90%")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_xticks(component_indices)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print_explanation(
        "The 'scree plot' (left) usually drops sharply after the first "
        "component or two -- another instance of the same pattern we saw "
        "with SVD singular values: real-world data tends to be dominated "
        "by a small number of directions, with the rest being minor "
        "corrections or noise."
    )

    pause()
    return cumulative


# =============================================================================
# CHAPTER 9 : PROJECTING DATA ONTO PRINCIPAL COMPONENTS
# =============================================================================

def project_data(X_centered: np.ndarray, eigenvectors: np.ndarray, k: int) -> np.ndarray:
    """Project centered data onto the first k principal components.

    Formula: Z = X_centered @ eigenvectors[:, :k]

    Each row of Z is the sample's coordinates in the new, rotated basis
    defined by the top-k principal components -- this is a simple
    dot-product ("how far along this direction is this point?") repeated
    for each of the k chosen directions.
    """
    k = max(1, min(k, eigenvectors.shape[1]))
    Vk = eigenvectors[:, :k]
    Z = X_centered @ Vk
    return Z


def chapter9_project_and_visualize(X_centered: np.ndarray, eigenvectors: np.ndarray,
                                     eigenvalues: np.ndarray, mean: np.ndarray, name: str) -> None:
    """CHAPTER 9: project the 2D data and draw the PC axes as arrows."""
    print_header(f"CHAPTER 9 : PROJECTING DATA ONTO PRINCIPAL COMPONENTS ({name})")

    print_explanation(
        "Projecting means asking, for every principal component "
        "direction, 'how far along this direction does each data point "
        "sit?' Mathematically this is just a dot product between the "
        "centered data and each eigenvector, done all at once via matrix "
        "multiplication: Z = X_centered @ eigenvectors[:, :k]."
    )

    Z_full = project_data(X_centered, eigenvectors, eigenvectors.shape[1])

    print_subheader("First 5 projected samples (new PC coordinates)")
    with np.printoptions(precision=4, suppress=True):
        print(Z_full[:5])

    if X_centered.shape[1] == 2:
        original_data = X_centered + mean
        scale = 2.5 * np.sqrt(eigenvalues)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(original_data[:, 0], original_data[:, 1], s=15, alpha=0.5, color="tab:blue")
        colors_arrows = ["tab:red", "tab:orange"]
        for i in range(2):
            direction = eigenvectors[:, i] * scale[i]
            ax.annotate(
                "", xy=(mean[0] + direction[0], mean[1] + direction[1]), xytext=mean,
                arrowprops=dict(arrowstyle="->", color=colors_arrows[i], linewidth=2.5),
            )
            ax.text(mean[0] + direction[0] * 1.1, mean[1] + direction[1] * 1.1,
                     f"PC{i + 1}", color=colors_arrows[i], fontsize=12, fontweight="bold")

        ax.scatter(*mean, color="black", marker="x", s=100, label="Mean")
        ax.set_title("Principal Component Directions on the Original Data")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

        print_explanation(
            "The red arrow (PC1) points along the direction of greatest "
            "spread -- notice it aligns with the diagonal stretch we built "
            "into the synthetic clusters. The orange arrow (PC2) is "
            "perpendicular to PC1 and captures the remaining, much "
            "smaller, spread."
        )

    pause()


# =============================================================================
# CHAPTER 10 : DIMENSIONALITY REDUCTION IN ACTION
# =============================================================================

def chapter10_dimensionality_reduction(data_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CHAPTER 10: reduce the 3D dataset down to 2D and 1D and visualize it."""
    print_header("CHAPTER 10 : DIMENSIONALITY REDUCTION IN ACTION (3D -> 2D -> 1D)")

    print_explanation(
        "Now for the real payoff: we take our 3-feature dataset and "
        "compress it down to 2 features, and then down to just 1 feature, "
        "using only the top-k principal components. Because Dataset B was "
        "built to lie near a 2D plane, reducing to 2 components should "
        "keep almost all of the meaningful structure."
    )

    X_centered, mean = center_data(data_3d)
    covariance = compute_covariance_matrix(X_centered)
    eigenvalues, eigenvectors = eigen_decomposition(covariance)
    ratios = explained_variance_ratio(eigenvalues) * 100

    print_subheader("Explained Variance per Component (3D dataset)")
    for i, r in enumerate(ratios, start=1):
        print(f"  PC{i}: {ascii_progress_bar(r)}")

    Z2 = project_data(X_centered, eigenvectors, 2)
    Z1 = project_data(X_centered, eigenvectors, 1)

    fig = plt.figure(figsize=(15, 5))

    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], s=12, alpha=0.6, color="tab:orange")
    ax0.set_title("Original (3 features)")
    ax0.set_xlabel("F1"); ax0.set_ylabel("F2"); ax0.set_zlabel("F3")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.scatter(Z2[:, 0], Z2[:, 1], s=12, alpha=0.6, color="tab:green")
    ax1.set_title("Reduced to 2 Principal Components")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.scatter(Z1[:, 0], np.zeros_like(Z1[:, 0]), s=12, alpha=0.6, color="tab:red")
    ax2.set_title("Reduced to 1 Principal Component")
    ax2.set_xlabel("PC1")
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print_explanation(
        "Going from 3D to 2D barely changes the shape of the data cloud "
        "at all -- that's because almost all of the variance lives in the "
        "first two components. Going all the way down to 1D flattens the "
        "cloud onto a single line, discarding the second dimension's "
        "spread entirely; this is a much bigger simplification, and you "
        "can see the cloud lose its width."
    )

    pause()
    return X_centered, eigenvalues, eigenvectors


# =============================================================================
# CHAPTER 11 : RECONSTRUCTION FROM k COMPONENTS
# =============================================================================

def reconstruct_data(X_centered: np.ndarray, eigenvectors: np.ndarray, mean: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct an approximation of the original data from k components.

    Steps:
        1. Project onto the top-k components:  Z = X_centered @ Vk        (n x k)
        2. Map back into the original feature space: X_approx = Z @ Vk.T (n x d)
        3. Add the mean back to undo the centering from Chapter 3.

    Because eigenvectors are orthonormal, Vk @ Vk.T acts as a projection
    matrix onto the k-dimensional subspace spanned by the top components --
    using fewer components means information outside that subspace is
    permanently discarded.
    """
    k = max(1, min(k, eigenvectors.shape[1]))
    Vk = eigenvectors[:, :k]
    Z = X_centered @ Vk               # (n x k)  -- project down
    X_approx = Z @ Vk.T + mean        # (n x d)  -- map back up + un-center
    return X_approx


def compute_reconstruction_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between original and reconstructed data."""
    return float(np.mean((original - reconstructed) ** 2))


def chapter11_reconstruction(data_3d: np.ndarray, X_centered: np.ndarray,
                               eigenvectors: np.ndarray, mean: np.ndarray) -> None:
    """CHAPTER 11: walk through the reconstruction formula with real shapes."""
    print_header("CHAPTER 11 : RECONSTRUCTING DATA FROM k COMPONENTS")

    print_explanation(
        "Reduction is only half the story -- we can also map reduced data "
        "back into the original feature space to see how much was lost. "
        "The formula is: X_approx = (X_centered @ Vk) @ Vk.T + mean. The "
        "first multiplication projects down to k dimensions; the second "
        "maps back up to d dimensions along the same k directions; adding "
        "the mean undoes the centering from Chapter 3."
    )

    n, d = data_3d.shape
    example_k = 2

    print_subheader(f"Example: reconstructing with k = {example_k}")
    print(f"  Z  shape       : ({n}, {example_k})   <- projected coordinates")
    print(f"  Vk.T shape     : ({example_k}, {d})   <- map back to original space")
    print(f"  X_approx shape : ({n}, {d})   <- same shape as the original data")

    reconstructed = reconstruct_data(X_centered, eigenvectors, mean, example_k)
    mse = compute_reconstruction_mse(data_3d, reconstructed)
    print(f"\n  Reconstruction MSE at k={example_k}: {mse:.5f}")

    pause()


# =============================================================================
# CHAPTER 12 : RECONSTRUCTION ERROR VS NUMBER OF COMPONENTS
# =============================================================================

def chapter12_error_curve(data_3d: np.ndarray, X_centered: np.ndarray,
                            eigenvectors: np.ndarray, mean: np.ndarray) -> None:
    """CHAPTER 12: show reconstruction error shrinking as k grows."""
    print_header("CHAPTER 12 : RECONSTRUCTION ERROR VS. NUMBER OF COMPONENTS")

    print_explanation(
        "Let's automatically reconstruct the dataset using every possible "
        "value of k, and plot how the reconstruction error (Mean Squared "
        "Error) shrinks as we add more components. By definition, using "
        "all components (k = number of features) gives PERFECT "
        "reconstruction with zero error."
    )

    d = data_3d.shape[1]
    k_values = list(range(1, d + 1))
    mse_values = []
    for k in k_values:
        reconstructed = reconstruct_data(X_centered, eigenvectors, mean, k)
        mse = compute_reconstruction_mse(data_3d, reconstructed)
        mse_values.append(mse)
        print(f"  k = {k} -> Reconstruction MSE: {mse:.6f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, mse_values, marker="o", color="tab:red", linewidth=2)
    ax.set_title("Reconstruction Error vs. Number of Components Kept")
    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print_explanation(
        "Notice the steep drop from k=1 to k=2, followed by a plunge to "
        "exactly zero at k=3 (the full number of features). This mirrors "
        "the cumulative energy curve from SVD image compression -- most "
        "of the 'recoverable' structure is captured by just the first "
        "couple of components."
    )

    pause()


# =============================================================================
# CHAPTER 13 : INTERACTIVE MODE
# =============================================================================

def chapter13_interactive_mode(data_3d: np.ndarray, X_centered: np.ndarray,
                                 eigenvectors: np.ndarray, eigenvalues: np.ndarray,
                                 mean: np.ndarray) -> None:
    """CHAPTER 13: let the user pick k interactively and see results live."""
    print_header("CHAPTER 13 : INTERACTIVE RECONSTRUCTION MODE")

    print_explanation(
        "Your turn. Type any integer k (1 to 3) to reduce and reconstruct "
        "the 3D dataset using that many principal components, and see the "
        "variance retained and reconstruction error update live. Type "
        "'exit' when you're done exploring."
    )

    max_k = eigenvectors.shape[1]
    cumulative = np.cumsum(explained_variance_ratio(eigenvalues)) * 100

    while True:
        try:
            user_input = input(colored(f"\nEnter k (1-{max_k}) or 'exit': ", Colors.BOLD)).strip()
        except EOFError:
            print(colored("\nNo more input available. Exiting interactive mode.", Colors.YELLOW))
            break

        if user_input.lower() == "exit":
            print(colored("Exiting interactive mode.", Colors.YELLOW))
            break

        try:
            k = int(user_input)
        except ValueError:
            print(colored("Please enter a valid integer, or 'exit'.", Colors.RED))
            continue

        if k < 1 or k > max_k:
            print(colored(f"k must be between 1 and {max_k}.", Colors.RED))
            continue

        reconstructed = reconstruct_data(X_centered, eigenvectors, mean, k)
        mse = compute_reconstruction_mse(data_3d, reconstructed)
        variance_retained = cumulative[k - 1]

        print_subheader(f"Results for k = {k}")
        print(f"  Variance Retained  : {ascii_progress_bar(variance_retained)}")
        print(f"  Reconstruction MSE : {mse:.6f}")

        fig = plt.figure(figsize=(11, 5))
        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax0.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], s=12, alpha=0.5, color="tab:blue", label="Original")
        ax0.set_title("Original Data")
        ax0.set_xlabel("F1"); ax0.set_ylabel("F2"); ax0.set_zlabel("F3")

        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        ax1.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                     s=12, alpha=0.5, color="tab:red", label=f"Reconstructed (k={k})")
        ax1.set_title(f"Reconstructed with k={k}")
        ax1.set_xlabel("F1"); ax1.set_ylabel("F2"); ax1.set_zlabel("F3")

        plt.tight_layout()
        plt.show()


# =============================================================================
# CHAPTER 14 : TERMINAL EXPLANATION OF DISCARDED COMPONENTS
# =============================================================================

def chapter14_explain_discard(k: int, total: int) -> None:
    """CHAPTER 14: print a standard explanation of what gets discarded at rank k."""
    print_header("CHAPTER 14 : WHAT HAPPENS TO THE DISCARDED COMPONENTS?")

    discarded = total - k
    print("=" * 50)
    print(f"Keeping first {k} principal component(s).")
    print(f"The remaining {discarded} component(s) are discarded.")
    print("These discarded directions mostly contain")
    print("small variations and noise.")
    print("The reconstructed data is an approximation.")
    print("=" * 50)

    print_explanation(
        "This is exactly what reconstruct_data(X_centered, eigenvectors, "
        "mean, k) does internally every time you call it. Because "
        "components are sorted by eigenvalue (variance explained), "
        "discarding the tail always throws away the *least* informative "
        "directions first."
    )

    pause()


# =============================================================================
# CHAPTER 15 : ANIMATION
# =============================================================================

def chapter15_animation(data_3d: np.ndarray, X_centered: np.ndarray,
                          eigenvectors: np.ndarray, mean: np.ndarray) -> None:
    """CHAPTER 15: animate the reconstruction as k increases from 1 to full."""
    print_header("CHAPTER 15 : WATCHING THE RECONSTRUCTION IMPROVE")

    print_explanation(
        "This animation reconstructs the 3D dataset at increasing k, "
        "pausing about one second between frames, so you can watch the "
        "flattened 'line' or 'plane' approximation grow back into the "
        "full, original data cloud as more components are added."
    )

    max_k = eigenvectors.shape[1]

    plt.ion()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    for k in range(1, max_k + 1):
        reconstructed = reconstruct_data(X_centered, eigenvectors, mean, k)
        ax.cla()
        ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                    s=12, alpha=0.6, color="tab:purple")
        ax.set_title(f"Reconstruction with k = {k} of {max_k} components")
        ax.set_xlabel("F1"); ax.set_ylabel("F2"); ax.set_zlabel("F3")
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(f"  Showing reconstruction at k = {k} ...")
        time.sleep(1.0)

    plt.ioff()
    plt.show()

    pause()


# =============================================================================
# CHAPTER 16 : MATH RECAP, STATISTICS, AND FINAL SUMMARY
# =============================================================================

def chapter16_math_recap() -> None:
    """Print an ASCII-diagram recap of the PCA equations."""
    print_header("CHAPTER 16A : THE MATHEMATICS BEHIND PCA")

    diagram = r"""
    Step 1: Center the data
        X_centered = X - mean(X)

    Step 2: Build the covariance matrix
        Cov = (X_centered^T @ X_centered) / (n - 1)          (d x d)

    Step 3: Eigen-decompose the covariance matrix
        Cov @ v_i = lambda_i * v_i        for each eigenvector v_i

    Step 4 (equivalently): Run SVD directly on the centered data
        X_centered = U @ Sigma @ V^T
        lambda_i = (sigma_i^2) / (n - 1)          <-- same eigenvalues!
        v_i = i-th row of V^T                      <-- same eigenvectors!

    Step 5: Project onto the top-k components
        Z = X_centered @ V[:, :k]                  (n x k)

    Step 6: Reconstruct an approximation
        X_approx = Z @ V[:, :k].T + mean(X)         (n x d)
    """
    print(diagram)

    print_explanation(
        "Every chapter in this program was one of these six steps, run on "
        "real synthetic data. Steps 1-3 use the covariance-matrix route; "
        "Step 4 shows the SVD route gives identical answers -- the same "
        "duality explored in the SVD Image Compression Visualizer, just "
        "applied to tabular data instead of pixels."
    )

    pause()


def chapter16_statistics(data_2d: np.ndarray, data_3d: np.ndarray,
                           eig_2d: np.ndarray, eig_3d: np.ndarray) -> None:
    """Print a summary of numerical statistics about both datasets."""
    print_header("CHAPTER 16B : DATASET STATISTICS")

    for name, data, eigenvalues in [("Dataset A (2D)", data_2d, eig_2d),
                                      ("Dataset B (3D)", data_3d, eig_3d)]:
        print_subheader(name)
        n, d = data.shape
        total_variance = float(np.sum(eigenvalues))
        print(f"  Samples                 : {n}")
        print(f"  Features                : {d}")
        print(f"  Number of components    : {len(eigenvalues)}")
        print(f"  Largest eigenvalue      : {eigenvalues[0]:.4f}")
        print(f"  Smallest eigenvalue     : {eigenvalues[-1]:.4f}")
        print(f"  Total variance (trace)  : {total_variance:.4f}")

    pause()


def chapter16_summary() -> None:
    """A big-picture wrap-up connecting PCA to the wider world."""
    print_header("CHAPTER 16C : SUMMARY -- WHY THIS ALL MATTERS")

    sections = [
        ("Why PCA Works",
         "PCA works because it finds the rotation of the coordinate axes "
         "that makes the data's spread as simple as possible to describe: "
         "aligned with the axes themselves, with each axis independently "
         "capturing as much variance as possible."),

        ("Why Dimensionality Reduction Works",
         "Real-world features are rarely independent -- they're "
         "correlated, redundant, or noisy. PCA identifies the few "
         "directions that carry almost all the meaningful variation, so "
         "dropping the rest loses little information while shrinking the "
         "data dramatically."),

        ("Why Eigenvalues Decrease",
         "Just like singular values in image compression, eigenvalues "
         "decrease because structured, real-world data is dominated by a "
         "handful of strong patterns, with the remaining directions "
         "contributing only minor corrections or noise."),

        ("Connection to SVD",
         "PCA's eigenvectors of the covariance matrix are exactly the "
         "right singular vectors of the centered data matrix, and its "
         "eigenvalues are the squared singular values divided by (n-1) -- "
         "PCA and SVD are the same tool wearing two different hats."),

        ("Whitening",
         "Dividing each principal component's projected values by the "
         "square root of its eigenvalue produces 'whitened' data with "
         "unit variance in every direction -- a common preprocessing step "
         "before many machine learning algorithms."),

        ("Connection to Machine Learning",
         "PCA is a standard preprocessing step to fight the curse of "
         "dimensionality, speed up training, reduce overfitting, and "
         "visualize high-dimensional datasets in 2D or 3D."),

        ("Connection to Autoencoders",
         "A linear autoencoder with a bottleneck layer, trained to "
         "reconstruct its input, learns to span the exact same subspace "
         "as PCA -- PCA can be thought of as the simplest possible neural "
         "network compressor."),

        ("Connection to Clustering & Visualization",
         "Projecting high-dimensional data onto its top 2 or 3 principal "
         "components is one of the most common ways to visualize "
         "clusters, outliers, and structure that would otherwise be "
         "invisible in raw high-dimensional space."),

        ("Connection to Genomics & Finance",
         "PCA is used to find dominant patterns of gene expression across "
         "thousands of genes, and to find dominant 'risk factors' driving "
         "correlated movements across large baskets of financial assets."),
    ]

    for title, body in sections:
        print_subheader(title)
        print_explanation(body)

    print("\n" + colored("=" * 79, Colors.CYAN))
    print(colored("Thank you for exploring PCA!".center(79), Colors.BOLD + Colors.GREEN))
    print(colored("=" * 79, Colors.CYAN))


# =============================================================================
# BONUS : MATPLOTLIB SLIDER-BASED INTERACTIVE VIEWER
# =============================================================================

def bonus_slider_viewer(data_3d: np.ndarray, X_centered: np.ndarray,
                          eigenvectors: np.ndarray, mean: np.ndarray) -> None:
    """A bonus interactive matplotlib widget: drag a slider to change k live."""
    print_header("BONUS : INTERACTIVE SLIDER VIEWER")
    print_explanation(
        "Drag the slider to change k in real time and watch the "
        "reconstructed 3D point cloud update instantly. Close the window "
        "when you're done."
    )

    max_k = eigenvectors.shape[1]
    start_k = 1

    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.2)

    reconstructed = reconstruct_data(X_centered, eigenvectors, mean, start_k)
    scatter = ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                           s=12, alpha=0.6, color="tab:purple")
    ax.set_title(f"k = {start_k}")

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.04])
    k_slider = Slider(slider_ax, "k", 1, max_k, valinit=start_k, valstep=1)

    def on_change(val):
        """Callback fired every time the slider moves."""
        k = int(k_slider.val)
        new_reconstruction = reconstruct_data(X_centered, eigenvectors, mean, k)
        scatter._offsets3d = (new_reconstruction[:, 0], new_reconstruction[:, 1], new_reconstruction[:, 2])
        ax.set_title(f"k = {k}")
        fig.canvas.draw_idle()

    k_slider.on_changed(on_change)
    plt.show()


# =============================================================================
# MAIN PROGRAM FLOW
# =============================================================================

def print_welcome_banner() -> None:
    """Print a friendly banner when the program starts."""
    banner = r"""
  _____   _____                _____                            
 |  __ \ / ____|  /\          / ____|                            
 | |__) | |      /  \        | (___   ___ ___  _ __   ___   _ __ 
 |  ___/| |     / /\ \        \___ \ / __/ _ \| '_ \ / _ \ | '__|
 | |    | |____/ ____ \       ____) | (_| (_) | |_) |  __/_| |   
 |_|     \_____/_/    \_\     |_____/ \___\___/| .__/ \___(_)_|   
                                                | |                
                                                |_|  PCA VISUALIZER
"""
    print(colored(banner, Colors.HEADER))
    print_explanation(
        "Welcome! This program teaches Principal Component Analysis (PCA) "
        "step-by-step, built entirely from scratch with numpy's linear "
        "algebra tools -- no sklearn, no scipy. Work through each chapter "
        "at your own pace -- press ENTER when prompted to move forward."
    )


def main() -> None:
    """Main entry point: runs every chapter in order."""
    print_welcome_banner()
    pause()

    # ---- Chapter 1 : generate data --------------------------------------
    data_2d, data_3d = chapter1_what_is_data()

    # ---- Chapter 2 : print matrices --------------------------------------
    chapter2_print_matrix(data_2d, "Dataset A (2D)")
    chapter2_print_matrix(data_3d, "Dataset B (3D)")

    # ---- Chapters 3-9 : full PCA pipeline on the 2D dataset --------------
    centered_2d, mean_2d = chapter3_mean_centering(data_2d, "Dataset A (2D)")
    cov_2d = chapter4_covariance_matrix(centered_2d, "Dataset A (2D)")
    eig_2d, vec_2d = chapter5_eigen_decomposition(cov_2d, "Dataset A (2D)")
    chapter6_connection_to_svd(centered_2d, eig_2d, vec_2d, "Dataset A (2D)")
    chapter7_interpret_components(eig_2d, "Dataset A (2D)")
    chapter8_explained_variance(eig_2d, "Dataset A (2D)")
    chapter9_project_and_visualize(centered_2d, vec_2d, eig_2d, mean_2d, "Dataset A (2D)")

    # ---- Chapters 10-15 : dimensionality reduction on the 3D dataset -----
    centered_3d, eig_3d, vec_3d = chapter10_dimensionality_reduction(data_3d)
    mean_3d = np.mean(data_3d, axis=0)
    chapter11_reconstruction(data_3d, centered_3d, vec_3d, mean_3d)
    chapter12_error_curve(data_3d, centered_3d, vec_3d, mean_3d)

    demo_k = 2
    chapter14_explain_discard(demo_k, vec_3d.shape[1])

    chapter13_interactive_mode(data_3d, centered_3d, vec_3d, eig_3d, mean_3d)
    chapter15_animation(data_3d, centered_3d, vec_3d, mean_3d)

    # ---- Chapter 16 : recap, statistics, summary --------------------------
    chapter16_math_recap()
    chapter16_statistics(data_2d, data_3d, eig_2d, eig_3d)
    chapter16_summary()

    # ---- Bonus : slider viewer ---------------------------------------------
    try:
        answer = input(colored("\nOpen the bonus interactive slider viewer? (y/n): ", Colors.BOLD)).strip().lower()
    except EOFError:
        answer = "n"
    if answer == "y":
        bonus_slider_viewer(data_3d, centered_3d, vec_3d, mean_3d)

    print(colored("\nProgram finished. Goodbye!\n", Colors.GREEN + Colors.BOLD))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n\nInterrupted by user. Goodbye!\n", Colors.RED))
        sys.exit(0)