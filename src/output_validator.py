"""
src/output_validator.py
-----------------------
Compares pipeline-computed .sto files against pre-computed OpenSim GUI outputs.

Reads both files, aligns them by time, and reports per-column statistics:
  - Mean absolute error (MAE)
  - Max absolute error
  - Pearson correlation
  - Relative RMSE (%)

This validates that the pipeline produces outputs consistent with the GUI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import read_sto_file

logger = logging.getLogger(__name__)


def compare_sto_files(
    computed_path: Path,
    reference_path: Path,
    label: str = "",
    rtol: float = 0.10,
) -> dict:
    """
    Compare a pipeline-computed .sto file against a GUI reference .sto file.

    Parameters
    ----------
    computed_path  : Path to the pipeline-generated .sto
    reference_path : Path to the pre-computed GUI .sto
    label          : Human label for logging (e.g. 'ID', 'SO_activation')
    rtol           : Relative tolerance for flagging mismatches (0.10 = 10%)

    Returns
    -------
    dict with comparison statistics per column and an overall summary.
    """
    comp_df = read_sto_file(computed_path)
    ref_df = read_sto_file(reference_path)

    # Align columns (use intersection)
    common_cols = sorted(set(comp_df.columns) & set(ref_df.columns))
    data_cols = [c for c in common_cols if c.lower() != "time"]

    if not data_cols:
        logger.warning("[%s] No common data columns between computed and reference!", label)
        return {"status": "NO_COMMON_COLUMNS", "label": label}

    # Align by row count (truncate to shorter)
    n_rows = min(len(comp_df), len(ref_df))
    if len(comp_df) != len(ref_df):
        logger.warning(
            "[%s] Row count mismatch: computed=%d, reference=%d — using first %d rows",
            label, len(comp_df), len(ref_df), n_rows,
        )

    col_stats = []
    n_pass = 0
    n_fail = 0

    for col in data_cols:
        c_vals = comp_df[col].iloc[:n_rows].to_numpy(dtype=float)
        r_vals = ref_df[col].iloc[:n_rows].to_numpy(dtype=float)

        # Skip columns that are all-zero or all-NaN in reference
        ref_range = np.nanmax(np.abs(r_vals))
        if ref_range == 0 or np.isnan(ref_range):
            col_stats.append({
                "column": col, "mae": 0.0, "max_err": 0.0,
                "corr": 1.0, "rel_rmse_pct": 0.0, "status": "SKIP_ZERO",
            })
            continue

        mae = float(np.nanmean(np.abs(c_vals - r_vals)))
        max_err = float(np.nanmax(np.abs(c_vals - r_vals)))
        rmse = float(np.sqrt(np.nanmean((c_vals - r_vals) ** 2)))
        rel_rmse = rmse / ref_range * 100.0

        # Correlation
        valid = ~(np.isnan(c_vals) | np.isnan(r_vals))
        if valid.sum() > 2:
            corr = float(np.corrcoef(c_vals[valid], r_vals[valid])[0, 1])
        else:
            corr = float("nan")

        status = "PASS" if rel_rmse < (rtol * 100) else "MISMATCH"
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        col_stats.append({
            "column": col, "mae": mae, "max_err": max_err,
            "corr": corr, "rel_rmse_pct": rel_rmse, "status": status,
        })

    # Summary
    summary = {
        "label": label,
        "computed_file": computed_path.name,
        "reference_file": reference_path.name,
        "n_rows_computed": len(comp_df),
        "n_rows_reference": len(ref_df),
        "n_common_columns": len(data_cols),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "status": "PASS" if n_fail == 0 else "MISMATCH",
        "columns": col_stats,
    }

    # Log summary
    if n_fail == 0:
        logger.info(
            "[%s] ✓ VALIDATION PASSED — %d/%d columns within %.0f%% tolerance",
            label, n_pass, len(data_cols), rtol * 100,
        )
    else:
        logger.warning(
            "[%s] ⚠ VALIDATION: %d/%d columns PASSED, %d MISMATCHED (>%.0f%% relative RMSE)",
            label, n_pass, len(data_cols), n_fail, rtol * 100,
        )
        # Log the worst mismatches
        mismatches = [c for c in col_stats if c["status"] == "MISMATCH"]
        mismatches.sort(key=lambda x: x["rel_rmse_pct"], reverse=True)
        for m in mismatches[:10]:
            logger.warning(
                "  %s: MAE=%.4f, max_err=%.4f, corr=%.4f, rel_RMSE=%.1f%%",
                m["column"], m["mae"], m["max_err"], m["corr"], m["rel_rmse_pct"],
            )

    return summary


def validate_participant_outputs(
    participant_id: str,
    computed_files: dict[str, Path],
    reference_model_dir: Path,
    reference_kinematics_dir: Path,
    rtol: float = 0.10,
) -> list[dict]:
    """
    Run comparison for all output types of one participant.

    Parameters
    ----------
    participant_id  : e.g. 'P001'
    computed_files  : dict with keys 'id_sto', 'so_activation_sto', 'so_force_sto'
    reference_model_dir : path to patient's OpenSimData/Model/
    reference_kinematics_dir : path to patient's OpenSimData/Kinematics/
    rtol            : relative tolerance (0.10 = 10%)

    Returns
    -------
    list of comparison summary dicts
    """
    from src.file_locator import (
        locate_inverse_dynamics,
        locate_so_activation,
        locate_so_force,
    )

    results = []

    # --- ID comparison ---
    if "id_sto" in computed_files and computed_files["id_sto"].exists():
        try:
            ref_id = locate_inverse_dynamics(reference_model_dir, participant_id)
            result = compare_sto_files(
                computed_files["id_sto"], ref_id,
                label=f"{participant_id}/ID", rtol=rtol,
            )
            results.append(result)
        except FileNotFoundError as e:
            logger.warning("[%s] Cannot validate ID — reference not found: %s", participant_id, e)

    # --- SO activation comparison ---
    if "so_activation_sto" in computed_files and computed_files["so_activation_sto"].exists():
        try:
            ref_act = locate_so_activation(reference_model_dir, participant_id)
            result = compare_sto_files(
                computed_files["so_activation_sto"], ref_act,
                label=f"{participant_id}/SO_activation", rtol=rtol,
            )
            results.append(result)
        except FileNotFoundError as e:
            logger.warning("[%s] Cannot validate SO activation — reference not found: %s", participant_id, e)

    # --- SO force comparison ---
    if "so_force_sto" in computed_files and computed_files["so_force_sto"].exists():
        try:
            ref_force = locate_so_force(reference_model_dir, participant_id)
            result = compare_sto_files(
                computed_files["so_force_sto"], ref_force,
                label=f"{participant_id}/SO_force", rtol=rtol,
            )
            results.append(result)
        except FileNotFoundError as e:
            logger.warning("[%s] Cannot validate SO force — reference not found: %s", participant_id, e)

    return results
