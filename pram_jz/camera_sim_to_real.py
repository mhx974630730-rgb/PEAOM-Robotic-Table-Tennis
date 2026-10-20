#!/usr/bin/env python3
"""Fit the affine perception model: observed_position = A @ true_position + b."""

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "calibration_data_accumulated_swapped.npz"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "linear_correction.npz"


def fit_affine_mapping(input_path: Path, output_path: Path):
    data = np.load(str(input_path), allow_pickle=False)
    required = {"true_positions", "triangulated_positions"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing arrays in {input_path}: {sorted(missing)}")

    true_positions = np.asarray(data["true_positions"], dtype=np.float64)
    observed_positions = np.asarray(
        data["triangulated_positions"], dtype=np.float64
    )
    if true_positions.shape != observed_positions.shape:
        raise ValueError("true_positions and triangulated_positions must have the same shape")
    if true_positions.ndim != 2 or true_positions.shape[1] != 3:
        raise ValueError("calibration arrays must have shape (N, 3)")
    if true_positions.shape[0] < 4:
        raise ValueError("at least four calibration samples are required")
    if not np.all(np.isfinite(true_positions)) or not np.all(np.isfinite(observed_positions)):
        raise ValueError("calibration arrays contain NaN or infinity")

    design = np.column_stack((true_positions, np.ones(len(true_positions))))
    solution, _, _, _ = np.linalg.lstsq(design, observed_positions, rcond=None)
    A = solution[:3, :].T
    b = solution[3, :]

    predicted = true_positions @ A.T + b
    residual = predicted - observed_positions
    rmse_xyz = np.sqrt(np.mean(residual ** 2, axis=0))
    rmse_3d = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, A=A, b=b)
    return A, b, rmse_xyz, rmse_3d


def main():
    parser = argparse.ArgumentParser(
        description="Fit the affine true-position-to-vision-position mapping."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    A, b, rmse_xyz, rmse_3d = fit_affine_mapping(
        args.input.resolve(), args.output.resolve()
    )
    print(f"Saved affine mapping to: {args.output.resolve()}")
    print("A =")
    print(A)
    print(f"b = {b}")
    print(f"residual RMSE xyz = {rmse_xyz} m")
    print(f"residual 3D RMSE = {rmse_3d:.8f} m")


if __name__ == "__main__":
    main()
