"""
Utility Functions for Patch Stitching & VQA Pipeline
=====================================================

Additional helper functions for image processing, visualization, and evaluation.
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def histogram_equalize(img):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for illumination normalization.
    
    Args:
        img (ndarray): Input BGR image (H×W×3, uint8)
    
    Returns:
        ndarray: Equalized image (H×W×3, uint8)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab_eq = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def histogram_match(src, ref):
    """
    Match histogram of src to ref (illumination normalization).
    
    Args:
        src (ndarray): Source image (H×W×3, uint8)
        ref (ndarray): Reference image (H×W×3, uint8)
    
    Returns:
        ndarray: Histogram-matched source image
    """
    # Simple per-channel linear normalization
    for ch in range(3):
        src_ch = src[:, :, ch].astype(np.float32)
        ref_ch = ref[:, :, ch].astype(np.float32)
        
        src_mean, src_std = src_ch.mean(), src_ch.std() + 1e-8
        ref_mean, ref_std = ref_ch.mean(), ref_ch.std() + 1e-8
        
        # Linear normalization: (x - src_mean) / src_std * ref_std + ref_mean
        src[:, :, ch] = np.clip(
            (src_ch - src_mean) / src_std * ref_std + ref_mean,
            0, 255
        ).astype(np.uint8)
    
    return src


def visualize_patch_matches(patch_a, patch_b, direction, title="Patch Comparison"):
    """
    Visualize edge matching between two patches (for debugging).
    
    Args:
        patch_a, patch_b (ndarray): Patches to compare
        direction (str): "horizontal" or "vertical"
        title (str): Window title
    
    Returns:
        ndarray: Side-by-side or stacked comparison image
    """
    if direction == "horizontal":
        # Stack horizontally
        vis = np.hstack([patch_a, patch_b])
    else:
        # Stack vertically
        vis = np.vstack([patch_a, patch_b])
    
    return vis


def compute_seam_quality(canvas, weight):
    """
    Compute quality metric for seams in stitched image.
    
    Args:
        canvas (ndarray): Stitched image
        weight (ndarray): Weight map (accumulation count per pixel)
    
    Returns:
        dict: Metrics including seam_confidence, coverage, etc.
    """
    seam_conf = float(np.mean(weight)) / float(np.max(weight) + 1e-6)
    coverage = float(np.sum(weight > 0)) / weight.size
    
    return {
        "seam_confidence": seam_conf,
        "coverage": coverage,
        "mean_weight": float(np.mean(weight)),
        "max_weight": float(np.max(weight)),
    }


def validate_submission_csv(csv_path):
    """
    Validate submission CSV format and contents.
    
    Args:
        csv_path (str): Path to submission.csv
    
    Returns:
        dict: Validation results with any errors or warnings
    """
    import pandas as pd
    
    result = {"valid": True, "errors": [], "warnings": []}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Could not read CSV: {e}")
        return result
    
    # Check required columns
    required_cols = {"id", "question_num", "option"}
    if not required_cols.issubset(df.columns):
        result["errors"].append(f"Missing columns. Expected: {required_cols}, Got: {set(df.columns)}")
        result["valid"] = False
    
    # Check option values
    valid_options = {1, 2, 3, 4, 5}
    invalid = set(df["option"].unique()) - valid_options
    if len(invalid) > 0:
        result["errors"].append(f"Invalid option values found: {invalid}. Expected: {valid_options}")
        result["valid"] = False
    
    # Warnings
    if len(df[df["option"] == 5]) > len(df) * 0.8:
        result["warnings"].append(f"Warning: {len(df[df['option'] == 5])} out of {len(df)} questions were skipped (option=5)")
    
    return result


def generate_synthetic_patches(base_image, rows, cols, overlap_ratio=0.2):
    """
    Generate synthetic patch dataset from a base image (for validation).
    
    Args:
        base_image (ndarray): Source image (H×W×3, uint8)
        rows, cols (int): Grid dimensions
        overlap_ratio (float): Fraction of patch to overlap
    
    Returns:
        list: List of individual patches
    """
    h, w = base_image.shape[:2]
    patch_h = h // rows
    patch_w = w // cols
    
    overlap_h = int(patch_h * overlap_ratio)
    overlap_w = int(patch_w * overlap_ratio)
    
    patches = []
    for r in range(rows):
        for c in range(cols):
            y = max(0, r * (patch_h - overlap_h))
            x = max(0, c * (patch_w - overlap_w))
            patch = base_image[y:y+patch_h, x:x+patch_w].copy()
            patches.append(patch)
    
    return patches


def measure_stitching_accuracy(ground_truth_map, stitched_map):
    """
    Estimate stitching accuracy (MSE between true and stitched).
    
    Args:
        ground_truth_map, stitched_map (ndarray): Images to compare
    
    Returns:
        float: Mean squared error
    """
    # Resize to match if necessary
    if ground_truth_map.shape != stitched_map.shape:
        h, w = ground_truth_map.shape[:2]
        stitched_map = cv2.resize(stitched_map, (w, h))
    
    mse = np.mean((ground_truth_map.astype(np.float32) - stitched_map.astype(np.float32)) ** 2)
    return mse


if __name__ == "__main__":
    print("[UTILS] Utility functions module loaded")
