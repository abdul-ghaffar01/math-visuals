#!/usr/bin/env python3
# =============================================================================
# SVD IMAGE COMPRESSION VISUALIZER
# -----------------------------------------------------------------------------
# An educational, single-file Python program that teaches Singular Value
# Decomposition (SVD) from first principles while using it to compress a
# real grayscale image.
#
# Run with:
#     python svd_visualizer.py
#
# Allowed libraries only: numpy, matplotlib, PIL (Pillow), os, plus a handful
# of standard-library modules (sys, time, textwrap) used purely for terminal
# presentation (no numerical work is done with them).
# =============================================================================

import os
import sys
import time
import textwrap
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image


# =============================================================================
# SECTION 0 : TERMINAL PRESENTATION HELPERS
# -----------------------------------------------------------------------------
# These helpers make the terminal output look "professional" -- section
# dividers, colored text (when the terminal supports ANSI escape codes),
# ASCII progress bars, and a consistent "press ENTER to continue" pause.
# None of these functions do any mathematics; they only format text.
# =============================================================================

class Colors:
    """ANSI escape codes for colored terminal text.

    If the terminal does not support ANSI colors (e.g. some Windows
    consoles), `supports_color()` will disable coloring gracefully so the
    program never crashes or prints garbage escape codes.
    """
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
    """Return True if the current terminal likely supports ANSI colors.

    We check that stdout is attached to a real terminal (not redirected to
    a file) and that we are not on a very old Windows console.
    """
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
        # If input is not available (e.g. running in a non-interactive
        # pipeline), just continue instead of crashing.
        pass


def ascii_progress_bar(percentage: float, width: int = 40) -> str:
    """Build a simple ASCII progress bar string for a given percentage.

    Example: ascii_progress_bar(72.5) -> "[#####################-------] 72.5%"
    """
    percentage = max(0.0, min(100.0, percentage))
    filled = int(round(width * percentage / 100.0))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {percentage:6.2f}%"


# =============================================================================
# SECTION 1 : IMAGE LOADING
# -----------------------------------------------------------------------------
# CHAPTER 1 explains that an image, mathematically, is just a matrix of
# numbers. This section is responsible for obtaining that matrix, either by
# loading a real file the user provides, or by generating a synthetic test
# image if no valid file is found. Only PIL, numpy and os are used here.
# =============================================================================

def generate_synthetic_image(size: int = 256) -> np.ndarray:
    """Generate a synthetic grayscale test image as a numpy array.

    We build the image out of a smooth gradient plus a few geometric shapes
    (a circle and stripes). This gives the image *structure* and
    *redundancy* -- exactly the property that makes SVD compression work
    well, and it means the demo works even if the user has no image file.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Smooth radial gradient (low-frequency, highly compressible content).
    gradient = 1.0 - np.sqrt(xx ** 2 + yy ** 2)
    gradient = np.clip(gradient, 0, 1)

    # A solid circle (adds a sharp edge -- higher-frequency content).
    circle_mask = (xx ** 2 + yy ** 2) < 0.25
    circle = np.zeros_like(xx)
    circle[circle_mask] = 1.0

    # Diagonal stripes (high-frequency content, harder to compress).
    stripes = 0.15 * np.sin(20 * (xx + yy))

    combined = 0.55 * gradient + 0.35 * circle + stripes
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    pixels = (combined * 255).astype(np.uint8)
    return pixels


def load_image(path: str = "") -> Tuple[np.ndarray, str]:
    """Load a grayscale image as a numpy matrix.

    If `path` points to a valid, readable image file, it is opened with
    PIL, converted to 8-bit grayscale ("L" mode), and returned as a numpy
    array. If the path is empty or invalid, a synthetic image is generated
    instead so the program always has something meaningful to work with.

    Returns:
        (matrix, source_description)
    """
    if path and os.path.isfile(path):
        try:
            img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
            matrix = np.array(img, dtype=np.float64)
            return matrix, f"loaded from file '{path}'"
        except Exception as exc:  # noqa: BLE001 - we want a friendly fallback
            print(colored(f"Could not open '{path}' ({exc}). "
                           f"Falling back to a synthetic image.", Colors.RED))

    synthetic = generate_synthetic_image(size=256).astype(np.float64)
    # Save it to disk so the user can see / reuse it later.
    out_path = "sample_generated.png"
    try:
        Image.fromarray(synthetic.astype(np.uint8)).save(out_path)
    except Exception:  # noqa: BLE001 - saving is a nice-to-have, not critical
        pass
    return synthetic, f"synthetically generated (saved to '{out_path}')"


# =============================================================================
# CHAPTER 1 : WHAT IS AN IMAGE, MATHEMATICALLY?
# =============================================================================

def chapter1_explain_image(matrix: np.ndarray, source: str) -> None:
    """Explain that a grayscale image is simply a 2D matrix of numbers."""
    print_header("CHAPTER 1 : WHAT IS AN IMAGE, MATHEMATICALLY?")

    height, width = matrix.shape
    total_pixels = height * width

    print_explanation(
        "A grayscale digital image is nothing more than a rectangular grid "
        "of numbers. Each number represents the brightness of one pixel, "
        "typically ranging from 0 (pure black) to 255 (pure white) for an "
        "8-bit image. If you strip away the idea of 'picture' entirely, "
        "what you are left with is a plain matrix A of shape (height x "
        "width). Every single tool we use in this program -- SVD included "
        "-- only ever sees this matrix. It has no notion of 'eyes', "
        "'edges', or 'objects'; it only sees numbers arranged in rows and "
        "columns."
    )

    print_subheader("Image Source")
    print(f"  Image {source}")

    print_subheader("Image Dimensions")
    print(f"  Height (rows)    : {height}")
    print(f"  Width  (columns) : {width}")
    print(f"  Total pixels     : {total_pixels}")

    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Original Grayscale Image ({height} x {width})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    pause()


# =============================================================================
# CHAPTER 2 : THE IMAGE AS A MATRIX
# =============================================================================

def chapter2_print_matrix(matrix: np.ndarray, crop_size: int = 8) -> None:
    """Print the raw pixel matrix (or a small crop of it if it's large)."""
    print_header("CHAPTER 2 : THE IMAGE MATRIX")

    print_explanation(
        "Every entry A[i, j] of the matrix is the brightness of the pixel "
        "at row i and column j. 'Rows' run top-to-bottom (the y-axis of "
        "the image) and 'columns' run left-to-right (the x-axis). A value "
        "close to 0 means a dark pixel; a value close to 255 means a "
        "bright pixel. Because real images are large (our example has "
        f"{matrix.shape[0] * matrix.shape[1]} pixels), we only print a "
        f"small {crop_size} x {crop_size} crop from the top-left corner so "
        "the numbers stay readable -- but remember, the SVD math below "
        "operates on the *entire* matrix, not just this crop."
    )

    height, width = matrix.shape
    rows_to_show = min(crop_size, height)
    cols_to_show = min(crop_size, width)
    crop = matrix[:rows_to_show, :cols_to_show]

    print_subheader(f"Top-left {rows_to_show} x {cols_to_show} crop of the pixel matrix")
    with np.printoptions(precision=0, suppress=True, linewidth=120):
        print(crop.astype(int))

    pause()


# =============================================================================
# CHAPTER 3 : COMPUTING THE SVD
# =============================================================================

def compute_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Singular Value Decomposition of `matrix`.

    For any matrix A of shape (m, n), SVD factors it as:

        A = U @ diag(S) @ VT

    where:
        U  is an (m x k) matrix of "left singular vectors" (orthonormal
           columns) -- they form a basis for the column space of A.
        S  is a 1D array of length k containing the "singular values",
           sorted from largest to smallest. They measure how much
           "stretching" happens along each corresponding direction.
        VT is a (k x n) matrix of "right singular vectors" (orthonormal
           rows) -- they form a basis for the row space of A.

    We use full_matrices=False (the "economy" SVD) so that k = min(m, n),
    which is exactly what we need for low-rank image reconstruction.
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    return U, S, VT


def chapter3_compute_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CHAPTER 3: Run the SVD and explain the shapes of U, S, VT."""
    print_header("CHAPTER 3 : SINGULAR VALUE DECOMPOSITION (SVD)")

    print_explanation(
        "SVD is a fundamental theorem of linear algebra: ANY real matrix A "
        "(no matter its shape or content) can be factored into three "
        "matrices: A = U * Sigma * V^T. Think of it as breaking a complex "
        "transformation into three simple steps: a rotation (V^T), a "
        "stretching along the axes (Sigma), and another rotation (U). "
        "For images, this means we can rebuild the picture out of a sum of "
        "simple rank-1 'building block' images, each weighted by a "
        "singular value."
    )

    print_subheader("Calling np.linalg.svd(image_matrix, full_matrices=False)")
    U, S, VT = compute_svd(matrix)

    print_subheader("Resulting Shapes")
    print(f"  U  shape : {U.shape}   -> left singular vectors (columns are basis vectors)")
    print(f"  S  shape : {S.shape}   -> singular values (a 1D list, sorted descending)")
    print(f"  VT shape : {VT.shape}   -> right singular vectors (rows are basis vectors)")

    print_explanation(
        "U's columns tell you the dominant 'vertical patterns' of the "
        "image (how brightness varies down the rows). VT's rows tell you "
        "the dominant 'horizontal patterns' (how brightness varies across "
        "the columns). S tells you how important each pattern pair is -- "
        "the larger the singular value, the more that pattern contributes "
        "to reconstructing the original image."
    )

    pause()
    return U, S, VT


# =============================================================================
# CHAPTER 4 : SINGULAR VALUES
# =============================================================================

def chapter4_singular_values(S: np.ndarray, matrix: np.ndarray) -> None:
    """Print the singular values and explain what they mean."""
    print_header("CHAPTER 4 : UNDERSTANDING THE SINGULAR VALUES")

    print_explanation(
        "Singular values measure the 'strength' or 'importance' of each "
        "pattern found by the SVD. They are always non-negative and sorted "
        "from largest to smallest. A large singular value means that "
        "pattern captures a lot of the image's energy (its overall "
        "structure); a small singular value means that pattern only "
        "contributes fine details or noise."
    )

    print_subheader("All Singular Values")
    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print(S)

    rank = int(np.sum(S > 1e-10))

    print_subheader("Key Statistics")
    print(f"  Largest singular value  : {S[0]:.4f}")
    print(f"  Smallest singular value : {S[-1]:.4f}")
    print(f"  Number of singular values: {len(S)}")
    print(f"  Rank of the matrix       : {rank}")

    print_explanation(
        "The 'rank' of the matrix is the number of non-zero (or "
        "numerically significant) singular values. A full-rank image "
        "has no exact redundancy, but in practice, most of the "
        "singular values are tiny compared to the first few -- meaning "
        "the image is *approximately* low rank, which is precisely why "
        "SVD compression works so well on real-world images."
    )

    pause()


# =============================================================================
# CHAPTER 5 : PLOTTING THE SINGULAR VALUES
# =============================================================================

def chapter5_plot_singular_values(S: np.ndarray) -> None:
    """Plot the singular values to visualize how quickly they decay."""
    print_header("CHAPTER 5 : VISUALIZING SINGULAR VALUE DECAY")

    print_explanation(
        "When we plot the singular values in order, we almost always see "
        "a steep drop at the start followed by a long, flat tail. This "
        "happens because natural images are dominated by smooth, "
        "low-frequency structure (captured by the first few singular "
        "values) with only a small amount of fine detail and noise "
        "(captured by the many small singular values at the tail). This "
        "'fast decay' is the mathematical reason that keeping only a "
        "handful of singular values still gives a recognizable image."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(S, color="tab:blue", linewidth=2)
    axes[0].set_title("Singular Values (linear scale)")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(S, color="tab:red", linewidth=2)
    axes[1].set_title("Singular Values (log scale)")
    axes[1].set_xlabel("Singular Value Index")
    axes[1].set_ylabel("Magnitude (log scale)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    pause()


# =============================================================================
# CHAPTER 6 : CUMULATIVE ENERGY
# =============================================================================

def compute_cumulative_energy(S: np.ndarray) -> np.ndarray:
    """Compute the cumulative percentage of 'energy' retained by the top-k
    singular values, for every k from 1 to len(S).

    Energy of a singular value is defined as sigma^2 (this comes from the
    Frobenius norm of the matrix, ||A||_F^2 = sum(sigma_i^2)). Cumulative
    energy percentage at k is:

        cumulative_energy(k) = 100 * sum(sigma_i^2 for i in 1..k) / sum(sigma_i^2 for all i)
    """
    energy = S ** 2
    total_energy = np.sum(energy)
    cumulative = np.cumsum(energy)
    cumulative_percentage = 100.0 * cumulative / total_energy
    return cumulative_percentage


def chapter6_cumulative_energy(S: np.ndarray) -> np.ndarray:
    """CHAPTER 6: explain and print cumulative energy retained at various k."""
    print_header("CHAPTER 6 : CUMULATIVE ENERGY")

    print_explanation(
        "'Energy' here is a linear-algebra concept, not a physical one: it "
        "is defined as the sum of the squares of the singular values, "
        "which equals the squared Frobenius norm of the image matrix. "
        "Cumulative energy tells us what percentage of the image's total "
        "'information content' is captured by using only the top-k "
        "singular values. Because the singular values decay fast, a "
        "surprisingly small k can already capture over 90% of the energy."
    )

    cumulative_percentage = compute_cumulative_energy(S)
    n = len(S)
    checkpoints = [1, 5, 10, 20, 50, 100, 200, 500]
    checkpoints = [k for k in checkpoints if k <= n]
    if n not in checkpoints:
        checkpoints.append(n)  # "All" singular values

    print_subheader("Energy Retained at Various k")
    for k in checkpoints:
        pct = cumulative_percentage[k - 1]
        label = "All" if k == n else f"Top {k:<4}"
        bar = ascii_progress_bar(pct)
        print(f"  {label} singular values -> Energy retained: {bar}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, n + 1), cumulative_percentage, color="tab:green", linewidth=2)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% energy")
    ax.axhline(99, color="black", linestyle="--", linewidth=1, label="99% energy")
    ax.set_title("Cumulative Energy vs. Number of Singular Values Kept")
    ax.set_xlabel("Number of Singular Values (k)")
    ax.set_ylabel("Cumulative Energy Retained (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    pause()
    return cumulative_percentage


# =============================================================================
# CHAPTER 7 : IMAGE RECONSTRUCTION FROM k SINGULAR VALUES
# =============================================================================

def reconstruct_image(U: np.ndarray, S: np.ndarray, VT: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct an approximation of the original image using only the
    first k singular values (a 'rank-k approximation').

    Steps:
        1. Take the first k columns of U      -> Uk  (shape: m x k)
        2. Take the first k singular values    -> Sk  (shape: k,)
        3. Take the first k rows of VT         -> VTk (shape: k x n)
        4. Reconstruct: A_k = Uk @ diag(Sk) @ VTk

    This works because the full reconstruction A = U @ diag(S) @ VT is a
    weighted sum of k rank-1 matrices (outer products u_i * s_i * v_i^T).
    Truncating to the first k terms keeps only the most important
    (highest-energy) patterns and discards the rest.
    """
    k = max(1, min(k, len(S)))  # clamp k to a valid range
    Uk = U[:, :k]           # (m x k)   -- keep first k columns
    Sk = np.diag(S[:k])     # (k x k)   -- diagonal matrix of top-k singular values
    VTk = VT[:k, :]         # (k x n)   -- keep first k rows

    # Matrix multiplication chain: (m x k) @ (k x k) @ (k x n) -> (m x n)
    reconstructed = Uk @ Sk @ VTk
    # Pixel values must stay within the valid [0, 255] range.
    reconstructed = np.clip(reconstructed, 0, 255)
    return reconstructed


def chapter7_explain_reconstruction(U: np.ndarray, S: np.ndarray, VT: np.ndarray) -> None:
    """CHAPTER 7: walk through the reconstruction formula with real shapes."""
    print_header("CHAPTER 7 : IMAGE RECONSTRUCTION FROM k SINGULAR VALUES")

    print_explanation(
        "Once we have U, S, and VT, we can rebuild an *approximation* of "
        "the original image using only the k most important singular "
        "values, instead of all of them. The formula is: "
        "A_k = U[:, :k] @ diag(S[:k]) @ VT[:k, :]. Geometrically, we are "
        "summing k rank-1 'layers', where each layer is the outer product "
        "of one column of U, one singular value, and one row of VT. The "
        "layers are added in order of importance, so the first few layers "
        "already look like a blurry version of the full image."
    )

    m, n = U.shape[0], VT.shape[1]
    example_k = min(10, len(S))

    print_subheader(f"Example: reconstructing with k = {example_k}")
    print(f"  Uk  shape  : ({m}, {example_k})   <- first {example_k} columns of U")
    print(f"  Sk  shape  : ({example_k}, {example_k})   <- diagonal matrix of top {example_k} singular values")
    print(f"  VTk shape  : ({example_k}, {n})   <- first {example_k} rows of VT")
    print(f"  Multiplication chain: ({m} x {example_k}) @ ({example_k} x {example_k}) @ ({example_k} x {n}) -> ({m} x {n})")

    pause()


# =============================================================================
# CHAPTER 8 & 9 : MULTI-RANK RECONSTRUCTION AND VISUAL GRID
# =============================================================================

def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between the original and reconstructed image."""
    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(mse: float, max_pixel: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio (in decibels), a common image-quality metric.

    PSNR = 10 * log10(MAX^2 / MSE). Higher is better; a lossless
    reconstruction has infinite PSNR (MSE = 0).
    """
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_pixel ** 2) / mse))


def compression_ratio(m: int, n: int, k: int) -> float:
    """Compute the compression ratio for a rank-k SVD approximation.

    Original storage    : m * n numbers (one per pixel)
    Compressed storage   : k * (m + n + 1) numbers
        - k columns of U, each of length m       -> k * m
        - k singular values                       -> k
        - k rows of VT, each of length n           -> k * n
      Total = k*m + k*n + k = k * (m + n + 1)

    Ratio = original / compressed. A ratio > 1 means we saved space.
    """
    original = m * n
    compressed = k * (m + n + 1)
    return original / compressed


def chapter8_reconstruct_multiple(
    U: np.ndarray, S: np.ndarray, VT: np.ndarray
) -> List[Tuple[int, np.ndarray]]:
    """CHAPTER 8: automatically reconstruct the image at several rank values."""
    print_header("CHAPTER 8 : RECONSTRUCTING AT MULTIPLE RANKS")

    print_explanation(
        "Let's watch the image improve as we keep more and more singular "
        "values. We will reconstruct the image using k = 1, 5, 10, 20, 50, "
        "100, 200, and the full rank, and display each result. Notice how "
        "quickly the image becomes recognizable, even though we are still "
        "discarding most of the singular values."
    )

    max_k = len(S)
    k_values = [k for k in [1, 5, 10, 20, 50, 100, 200] if k <= max_k]
    if max_k not in k_values:
        k_values.append(max_k)  # "Full Rank"

    results = []
    for k in k_values:
        reconstructed = reconstruct_image(U, S, VT, k)
        results.append((k, reconstructed))
        label = "Full Rank" if k == max_k else f"k = {k}"
        print(f"  Reconstructed image at {label} ... done")

    print_explanation(
        "Each reconstructed image above was produced by the same "
        "reconstruct_image(U, S, VT, k) function, just with a different k. "
        "The individual figures for each rank are shown next; a combined "
        "comparison grid follows in Chapter 9."
    )

    for k, reconstructed in results:
        label = "Full Rank" if k == max_k else f"k = {k}"
        plt.figure(figsize=(4, 4))
        plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Reconstruction ({label})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    pause()
    return results


def chapter9_display_grid(
    results: List[Tuple[int, np.ndarray]], S: np.ndarray, m: int, n: int
) -> None:
    """CHAPTER 9: display all reconstructions together in one large figure."""
    print_header("CHAPTER 9 : SIDE-BY-SIDE COMPARISON GRID")

    print_explanation(
        "Seeing all the reconstructions together makes the trade-off "
        "between compression and quality obvious: low k gives a tiny, "
        "blocky file but a blurry image; high k gives a larger file but "
        "an image nearly identical to the original."
    )

    cumulative_percentage = compute_cumulative_energy(S)
    max_k = len(S)

    n_images = len(results)
    cols = 4
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    for idx, (k, reconstructed) in enumerate(results):
        ratio = compression_ratio(m, n, k)
        energy = cumulative_percentage[k - 1]
        label = "Full" if k == max_k else str(k)

        ax = axes[idx]
        ax.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"Rank = {label}\nCompression = {ratio:.1f}x\nEnergy = {energy:.1f}%",
            fontsize=10,
        )
        ax.axis("off")

        print(f"  Rank = {label:<5} Compression = {ratio:6.2f}x   Energy = {energy:6.2f}%")

    # Hide any unused subplot axes.
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("SVD Image Reconstruction at Increasing Rank", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    pause()


# =============================================================================
# CHAPTER 10 : COMPRESSION RATIO MATHEMATICS
# =============================================================================

def chapter10_compression_ratio(m: int, n: int) -> None:
    """CHAPTER 10: explain the compression ratio formula with concrete numbers."""
    print_header("CHAPTER 10 : COMPRESSION RATIO MATHEMATICS")

    print_explanation(
        "Storing the original image means storing every pixel: m * n "
        "numbers. Storing a rank-k SVD approximation instead means storing "
        "only Uk (m*k numbers), Sk (k numbers), and VTk (k*n numbers) -- a "
        "total of k * (m + n + 1) numbers. As long as k is much smaller "
        "than m and n, this is dramatically less data, at the cost of "
        "some image quality."
    )

    original_numbers = m * n
    example_ks = [5, 20, 50, 100]

    print_subheader(f"Original image: {m} x {n} = {original_numbers} numbers stored")
    for k in example_ks:
        compressed_numbers = k * (m + n + 1)
        ratio = compression_ratio(m, n, k)
        print(f"  k = {k:<4} -> Compressed numbers stored: {compressed_numbers:<8} "
              f"Compression Ratio: {ratio:.2f}x")

    pause()


# =============================================================================
# CHAPTER 11 : INTERACTIVE MODE
# =============================================================================

def chapter11_interactive_mode(
    U: np.ndarray, S: np.ndarray, VT: np.ndarray, original: np.ndarray
) -> None:
    """CHAPTER 11: let the user pick k interactively and see the results live."""
    print_header("CHAPTER 11 : INTERACTIVE RECONSTRUCTION MODE")

    print_explanation(
        "Now it's your turn. Type any integer k to reconstruct the image "
        "using that many singular values, and see the compression ratio, "
        "energy retained, MSE, and PSNR update live. Type 'exit' when "
        "you're done exploring."
    )

    m, n = original.shape
    cumulative_percentage = compute_cumulative_energy(S)
    max_k = len(S)

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

        reconstructed = reconstruct_image(U, S, VT, k)
        ratio = compression_ratio(m, n, k)
        energy = cumulative_percentage[k - 1]
        mse = compute_mse(original, reconstructed)
        psnr = compute_psnr(mse)

        print_subheader(f"Results for k = {k}")
        print(f"  Compression Ratio : {ratio:.2f}x")
        print(f"  Energy Preserved  : {ascii_progress_bar(energy)}")
        print(f"  Mean Squared Error: {mse:.4f}")
        print(f"  PSNR              : {psnr:.2f} dB")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Reconstructed (k={k})")
        axes[1].axis("off")

        diff = np.abs(original - reconstructed)
        im = axes[2].imshow(diff, cmap="inferno")
        axes[2].set_title("Difference (|Original - Reconstructed|)")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# =============================================================================
# CHAPTER 12 : TERMINAL EXPLANATION OF DISCARDED SINGULAR VALUES
# =============================================================================

def chapter12_explain_discard(k: int, total: int) -> None:
    """CHAPTER 12: print a standard explanation of what gets discarded at rank k."""
    print_header("CHAPTER 12 : WHAT HAPPENS TO THE DISCARDED SINGULAR VALUES?")

    discarded = total - k
    print("=" * 50)
    print(f"Keeping first {k} singular values.")
    print(f"The remaining {discarded} singular values are ignored.")
    print("These discarded values mostly contain")
    print("small details and noise.")
    print("The reconstructed image is an approximation.")
    print("=" * 50)

    print_explanation(
        "This isn't a special case -- it is exactly what "
        "reconstruct_image(U, S, VT, k) does internally every time you "
        "call it. Because singular values are sorted from largest to "
        "smallest, discarding the tail always discards the *least* "
        "important information first, which is why low-rank "
        "approximations still look so close to the original."
    )

    pause()


# =============================================================================
# CHAPTER 13 : ANIMATION OF INCREASING RANK
# =============================================================================

def chapter13_animation(U: np.ndarray, S: np.ndarray, VT: np.ndarray) -> None:
    """CHAPTER 13: animate the image sharpening as k increases."""
    print_header("CHAPTER 13 : WATCHING THE IMAGE COME INTO FOCUS")

    print_explanation(
        "This animation reconstructs the image at increasing rank, "
        "pausing about one second between frames, so you can watch the "
        "image sharpen from a blurry blob into a near-perfect copy of the "
        "original as more singular values are added."
    )

    max_k = len(S)
    k_values = [k for k in [1, 5, 10, 20, 50, 100, 200] if k <= max_k]
    if max_k not in k_values:
        k_values.append(max_k)

    plt.ion()  # interactive mode so we can update one figure repeatedly
    fig, ax = plt.subplots(figsize=(5, 5))
    img_display = ax.imshow(np.zeros_like(U @ np.diag(S) @ VT), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")

    for k in k_values:
        reconstructed = reconstruct_image(U, S, VT, k)
        label = "Full Rank" if k == max_k else f"k = {k}"
        img_display.set_data(reconstructed)
        ax.set_title(f"Rank {label}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        print(f"  Showing reconstruction at {label} ...")
        time.sleep(1.0)

    plt.ioff()
    plt.show()

    pause()


# =============================================================================
# CHAPTER 14 : THE MATHEMATICS OF A = U * SIGMA * V^T
# =============================================================================

def chapter14_math_explanation() -> None:
    """CHAPTER 14: an ASCII-diagram explanation of the SVD equation."""
    print_header("CHAPTER 14 : THE MATHEMATICS BEHIND A = U SIGMA V^T")

    diagram = r"""
        A            =        U            Sigma           V^T
    (m x n)                (m x k)        (k x k)         (k x n)

    [ . . . . ]        [ . . . ]      [ s1  0  0 ]      [ . . . . ]
    [ . . . . ]   =    [ . . . ]  @   [  0 s2  0 ]  @    [ . . . . ]
    [ . . . . ]        [ . . . ]      [  0  0 s3 ]       [ . . . . ]
    [ . . . . ]        [ . . . ]

     original          rotation       stretching        rotation
      image             (left)      (scaling factors)    (right)
    """
    print(diagram)

    print_explanation("U -- Left Singular Vectors:")
    print_explanation(
        "  Each column of U is an orthonormal 'direction' in the space of "
        "image columns. You can think of U as describing reusable "
        "vertical brightness patterns that get combined to build every "
        "column of the image."
    )

    print_explanation("Sigma (S) -- Stretching Amounts:")
    print_explanation(
        "  The diagonal values of Sigma tell you how strongly each "
        "matching pair of patterns (one from U, one from VT) should be "
        "combined. Bigger singular value = that pattern pair matters more "
        "for reconstructing the image."
    )

    print_explanation("V^T -- Right Singular Vectors:")
    print_explanation(
        "  Each row of VT is an orthonormal 'direction' in the space of "
        "image rows -- reusable horizontal brightness patterns. Together, "
        "U, Sigma, and VT let us rebuild the exact original image (using "
        "all k values) or a compressed approximation (using fewer)."
    )

    pause()


# =============================================================================
# CHAPTER 15 : IMAGE / MATRIX STATISTICS
# =============================================================================

def chapter15_statistics(matrix: np.ndarray, U: np.ndarray, S: np.ndarray, VT: np.ndarray) -> None:
    """CHAPTER 15: print a summary of numerical statistics about the matrix."""
    print_header("CHAPTER 15 : IMAGE AND MATRIX STATISTICS")

    height, width = matrix.shape
    rank = int(np.sum(S > 1e-10))
    trace = float(np.trace(matrix)) if height == width else None
    frobenius_norm = float(np.linalg.norm(matrix, "fro"))

    print_subheader("Statistics")
    print(f"  Height                  : {height}")
    print(f"  Width                   : {width}")
    print(f"  Rank                    : {rank}")
    print(f"  Number of Singular Vals : {len(S)}")
    print(f"  Largest Singular Value  : {S[0]:.4f}")
    print(f"  Smallest Singular Value : {S[-1]:.4f}")
    if trace is not None:
        print(f"  Trace                   : {trace:.4f}")
    else:
        print(f"  Trace                   : N/A (matrix is not square, {height} != {width})")
    print(f"  Frobenius Norm          : {frobenius_norm:.4f}")

    print_explanation(
        "The Frobenius norm is the square root of the sum of squares of "
        "every entry in the matrix -- it is the direct matrix analogue of "
        "a vector's length, and it equals sqrt(sum(sigma_i^2)), tying "
        "these statistics straight back to the singular values themselves."
    )

    pause()


# =============================================================================
# CHAPTER 16 : FINAL SUMMARY
# =============================================================================

def chapter16_summary() -> None:
    """CHAPTER 16: a big-picture wrap-up connecting SVD to the wider world."""
    print_header("CHAPTER 16 : SUMMARY -- WHY THIS ALL MATTERS")

    sections = [
        ("Why SVD Works",
         "SVD works because it is a universal factorization: every real "
         "matrix, no matter how it was generated, can be decomposed into "
         "rotations and a stretch. This makes it a general-purpose tool "
         "for understanding the 'shape' of any data, not just images."),

        ("Why Compression Works",
         "Compression works because we don't need perfect precision to "
         "recognize an image -- discarding small singular values removes "
         "information the human eye barely notices, while keeping the "
         "structure it relies on most."),

        ("Why Singular Values Decrease",
         "Singular values decrease because natural signals (images, "
         "audio, sensor data) are rarely random -- they are smooth and "
         "structured, so a few dominant patterns explain most of the "
         "variation, leaving only small residual corrections."),

        ("Why Images Contain Redundancy",
         "Neighboring pixels tend to have similar brightness (a smooth "
         "sky, a uniform wall). This local similarity is redundancy, and "
         "redundancy is exactly what low-rank approximations exploit."),

        ("Connection to PCA",
         "Principal Component Analysis is mathematically SVD applied to a "
         "mean-centered data matrix; the 'principal components' are just "
         "the right singular vectors, ranked by singular value."),

        ("Connection to AI",
         "Modern AI systems are built on matrix and tensor operations. "
         "Understanding SVD gives you the intuition to understand rank, "
         "compression, and information content in far more complex "
         "models."),

        ("Connection to Machine Learning",
         "SVD underlies dimensionality reduction, denoising, and latent "
         "factor models used throughout classical machine learning "
         "pipelines."),

        ("Connection to Neural Network Compression",
         "Large neural network weight matrices can be compressed with "
         "low-rank factorizations very similar to what we did here, "
         "reducing model size and speeding up inference with minimal "
         "accuracy loss."),

        ("Connection to Recommendation Systems",
         "Classic recommender systems factor a giant, sparse user-item "
         "matrix into low-rank user and item factors using SVD-like "
         "techniques -- the same 'keep only the top-k patterns' idea we "
         "used to compress an image."),
    ]

    for title, body in sections:
        print_subheader(title)
        print_explanation(body)

    print("\n" + colored("=" * 79, Colors.CYAN))
    print(colored("Thank you for exploring SVD Image Compression!".center(79), Colors.BOLD + Colors.GREEN))
    print(colored("=" * 79, Colors.CYAN))


# =============================================================================
# BONUS : MATPLOTLIB SLIDER-BASED INTERACTIVE VIEWER
# =============================================================================

def bonus_slider_viewer(U: np.ndarray, S: np.ndarray, VT: np.ndarray, original: np.ndarray) -> None:
    """A bonus interactive matplotlib widget: drag a slider to change k live.

    This uses only matplotlib.widgets.Slider (no extra libraries) to let
    the user explore rank interactively with the mouse instead of typing.
    Close the window to return to the terminal program.
    """
    print_header("BONUS : INTERACTIVE SLIDER VIEWER")
    print_explanation(
        "Drag the slider at the bottom of the window to change k in real "
        "time and watch the reconstruction update instantly. Close the "
        "window when you're done."
    )

    max_k = len(S)
    m, n = original.shape
    start_k = min(20, max_k)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    plt.subplots_adjust(bottom=0.2)

    reconstructed = reconstruct_image(U, S, VT, start_k)
    img_display = ax.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ratio = compression_ratio(m, n, start_k)
    ax.set_title(f"k = {start_k}   Compression = {ratio:.1f}x")

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.04])
    k_slider = Slider(slider_ax, "k", 1, max_k, valinit=start_k, valstep=1)

    def on_change(val):
        """Callback fired every time the slider moves."""
        k = int(k_slider.val)
        new_img = reconstruct_image(U, S, VT, k)
        img_display.set_data(new_img)
        new_ratio = compression_ratio(m, n, k)
        ax.set_title(f"k = {k}   Compression = {new_ratio:.1f}x")
        fig.canvas.draw_idle()

    k_slider.on_changed(on_change)
    plt.show()


# =============================================================================
# MAIN PROGRAM FLOW
# =============================================================================

def print_welcome_banner() -> None:
    """Print a friendly banner when the program starts."""
    banner = r"""
   _____ __      ______     _____                                             
  / ____|\ \    / /  _ \   / ____|                                            
 | (___   \ \  / /| | | | | |     ___  _ __ ___  _ __  _ __ ___  ___ ___  ___ 
  \___ \   \ \/ / | | | | | |    / _ \| '_ ` _ \| '_ \| '__/ _ \/ __/ __|/ _ \
  ____) |   \  /  | |_| | | |___| (_) | | | | | | |_) | | |  __/\__ \__ \  __/
 |_____/     \/   |____/   \_____\___/|_| |_| |_| .__/|_|  \___||___/___/\___|
                                                  | |                          
                                                  |_|  IMAGE VISUALIZER        
"""
    print(colored(banner, Colors.HEADER))
    print_explanation(
        "Welcome! This program teaches Singular Value Decomposition (SVD) "
        "step-by-step, using real image compression as a hands-on example. "
        "Work through each chapter at your own pace -- press ENTER when "
        "prompted to move forward."
    )


def main() -> None:
    """Main entry point: runs every chapter in order."""
    print_welcome_banner()
    pause()

    # ---- Ask the user for an image path (optional) --------------------
    print_header("STEP 0 : CHOOSE AN IMAGE")
    print_explanation(
        "Enter the path to a grayscale (or color -- it will be converted) "
        "image file, or just press ENTER to use an automatically generated "
        "synthetic test image."
    )
    try:
        user_path = input(colored("\nImage path (or ENTER for synthetic image): ", Colors.BOLD)).strip()
    except EOFError:
        user_path = ""

    matrix, source = load_image(user_path)
    m, n = matrix.shape

    # ---- Chapters 1-6 : setup, matrix, SVD, singular values, energy ---
    chapter1_explain_image(matrix, source)
    chapter2_print_matrix(matrix)
    U, S, VT = chapter3_compute_svd(matrix)
    chapter4_singular_values(S, matrix)
    chapter5_plot_singular_values(S)
    chapter6_cumulative_energy(S)

    # ---- Chapters 7-10 : reconstruction, comparison grid, ratio -------
    chapter7_explain_reconstruction(U, S, VT)
    results = chapter8_reconstruct_multiple(U, S, VT)
    chapter9_display_grid(results, S, m, n)
    chapter10_compression_ratio(m, n)

    # ---- Chapter 12 (demonstrated with a concrete k before interactive) ---
    demo_k = min(20, len(S))
    chapter12_explain_discard(demo_k, len(S))

    # ---- Chapter 11 : interactive terminal mode ------------------------
    chapter11_interactive_mode(U, S, VT, matrix)

    # ---- Chapter 13 : animation -----------------------------------------
    chapter13_animation(U, S, VT)

    # ---- Chapter 14-16 : math recap, statistics, summary -----------------
    chapter14_math_explanation()
    chapter15_statistics(matrix, U, S, VT)
    chapter16_summary()

    # ---- Bonus : slider viewer (optional) ---------------------------------
    try:
        answer = input(colored("\nOpen the bonus interactive slider viewer? (y/n): ", Colors.BOLD)).strip().lower()
    except EOFError:
        answer = "n"
    if answer == "y":
        bonus_slider_viewer(U, S, VT, matrix)

    print(colored("\nProgram finished. Goodbye!\n", Colors.GREEN + Colors.BOLD))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n\nInterrupted by user. Goodbye!\n", Colors.RED))
        sys.exit(0)