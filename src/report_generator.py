"""
src/report_generator.py
-----------------------
MODULE 9: Automated summary report generation.

Computes descriptive statistics (median, IQR, min, max) for all biomechanical
variables across the full participant cohort and writes a formatted report.

Also validates pipeline output against the thesis-published values — flags any
variable that differs by more than 5% from the published cohort median.

No OpenSim dependency — pure pandas.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Thesis reference values (from Tables 1-3)
#  Used to validate pipeline output matches manual analysis.
# ─────────────────────────────────────────────────────────────

THESIS_MEDIANS: dict[str, float] = {
    # Joint Moments (Table 1, Nm)
    "lumbar_flexion_Nm":     43.0,
    "lumbar_sideflex_Nm":    12.0,
    "lumbar_rotation_Nm":     4.10,
    "arm_flexion_Nm":         4.36,
    "arm_abduction_Nm":       4.44,
    "arm_rotation_Nm":        1.93,
    "elbow_flexion_Nm":       1.78,
    "prosup_Nm":              0.091,
    # Joint Reaction Forces (Table 2, N)
    "acromion_humerus_N":  1551.0,
    "elbow_ulna_N":         403.0,
    "radioulnar_N":         220.0,
    "wrist_N":              122.0,
    "back_torso_N":          54.9,
}

VALIDATION_TOLERANCE = 0.05  # 5% tolerance


# ─────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────

def compute_descriptives(
    df: pd.DataFrame,
    variable_list: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for biomechanical variables.

    Parameters
    ----------
    df            : Master dataset DataFrame (one row per participant)
    variable_list : Subset of columns to describe. If None, uses all numeric columns.

    Returns
    -------
    pd.DataFrame with columns: variable, n, median, IQR, min, max, mean, std
    """
    if variable_list is None:
        variable_list = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude demographic numeric columns
        exclude = ["age", "height_m", "weight_kg", "work_hrs_per_week",
                   "scanning_hrs_per_week", "patients_per_day", "min_per_patient",
                   "years_experience", "probe_weight_g", "chair_height_cm"]
        variable_list = [v for v in variable_list if v not in exclude]

    rows = []
    for var in variable_list:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        if len(col) == 0:
            continue
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        rows.append({
            "variable": var,
            "n": len(col),
            "median": round(col.median(), 3),
            "IQR": round(q3 - q1, 3),
            "min": round(col.min(), 3),
            "max": round(col.max(), 3),
            "mean": round(col.mean(), 3),
            "std": round(col.std(), 3),
        })

    return pd.DataFrame(rows)


def compare_with_thesis_values(
    computed_df: pd.DataFrame,
    reference: dict[str, float] = THESIS_MEDIANS,
    tolerance: float = VALIDATION_TOLERANCE,
) -> pd.DataFrame:
    """
    Compare computed cohort medians against thesis-published values.
    Returns a report DataFrame flagging discrepancies > tolerance.

    Parameters
    ----------
    computed_df : Output from compute_descriptives()
    reference   : Dict mapping variable name → published median
    tolerance   : Fractional tolerance (0.05 = 5%)

    Returns
    -------
    pd.DataFrame with columns: variable, published_median, computed_median, pct_diff, status
    """
    report_rows = []
    for var, pub_median in reference.items():
        row_match = computed_df[computed_df["variable"] == var]
        if row_match.empty:
            report_rows.append({
                "variable": var,
                "published_median": pub_median,
                "computed_median": None,
                "pct_diff": None,
                "status": "MISSING — variable not in output",
            })
            continue

        computed_median = float(row_match["median"].iloc[0])
        if pub_median == 0:
            pct_diff = float("inf") if computed_median != 0 else 0.0
        else:
            pct_diff = abs(computed_median - pub_median) / abs(pub_median)

        status = "OK" if pct_diff <= tolerance else f"WARNING — {pct_diff:.1%} diff"
        report_rows.append({
            "variable": var,
            "published_median": pub_median,
            "computed_median": round(computed_median, 3),
            "pct_diff": f"{pct_diff:.2%}",
            "status": status,
        })

    return pd.DataFrame(report_rows)


def write_report(
    master_csv_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Generate and write the full summary report.

    Parameters
    ----------
    master_csv_path : Path to results/master_dataset.csv
    output_path     : Path to write results/summary_report.txt
    """
    master_csv_path = Path(master_csv_path)
    output_path = Path(output_path)

    if not master_csv_path.exists():
        raise FileNotFoundError(f"Master dataset not found: {master_csv_path}")

    df = pd.read_csv(master_csv_path)
    n_participants = len(df)
    logger.info("Generating report for %d participants", n_participants)

    # Compute descriptives
    descriptives = compute_descriptives(df)

    # Validate against thesis
    validation = compare_with_thesis_values(descriptives)

    # Build report text
    lines = _build_report_text(descriptives, validation, n_participants)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Summary report written → %s", output_path.name)

    # Print validation warnings to console
    warnings = validation[validation["status"].str.startswith("WARNING")]
    if not warnings.empty:
        logger.warning(
            "Validation: %d variable(s) differ from thesis by >5%%:\n%s",
            len(warnings), warnings[["variable", "published_median", "computed_median", "pct_diff"]].to_string(index=False)
        )
    else:
        logger.info("Validation: All values within 5%% of thesis-published medians. ✓")


# ─────────────────────────────────────────────────────────────
#  Report formatting
# ─────────────────────────────────────────────────────────────

def _build_report_text(
    descriptives: pd.DataFrame,
    validation: pd.DataFrame,
    n_participants: int,
) -> list[str]:
    """Build the full report as a list of text lines."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 72

    lines += [
        sep,
        "  OPENSIM BIOMECHANICAL PIPELINE — SUMMARY REPORT",
        f"  Generated: {now}",
        f"  Participants: {n_participants}",
        sep, "",
    ]

    # --- Section 1: Joint Moments (Table 1 equivalent) ---
    moment_vars = [v for v in descriptives["variable"] if v.endswith("_Nm")]
    if moment_vars:
        lines += ["─" * 72, "TABLE 1 — Net Joint Moments (RMS, Nm)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, moment_vars))
        lines.append("")

    # --- Section 2: Joint Reaction Forces (Table 2 equivalent) ---
    force_vars = [v for v in descriptives["variable"] if v.endswith("_N")]
    if force_vars:
        lines += ["─" * 72, "TABLE 2 — Joint Reaction Forces (RMS Resultant, N)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, force_vars))
        lines.append("")

    # --- Section 3: Muscle Activations (Table 3 equivalent) ---
    activation_vars = [v for v in descriptives["variable"] if v.endswith("_activation")]
    if activation_vars:
        lines += ["─" * 72, "TABLE 3 — Muscle Activations (Full-Wave Rectified RMS, 0-1)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, activation_vars))
        lines.append("")

    # --- Section 4: Joint Kinematics ---
    kin_vars = [v for v in descriptives["variable"] if v.endswith("_deg")]
    if kin_vars:
        lines += ["─" * 72, "TABLE 4 — Joint Kinematics (RMS, degrees)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, kin_vars))
        lines.append("")

    # --- Section 5: Validation ---
    lines += [
        "─" * 72,
        "VALIDATION — Comparison with Thesis Published Values (±5% tolerance)",
        "─" * 72,
    ]
    if not validation.empty:
        lines.append(validation.to_string(index=False))
    else:
        lines.append("  (No thesis reference values configured)")
    lines += ["", sep, "END OF REPORT", sep]

    return lines


def _format_descriptives_table(desc_df: pd.DataFrame, variables: list[str]) -> str:
    """Format a subset of descriptives as an aligned text table."""
    subset = desc_df[desc_df["variable"].isin(variables)].copy()
    subset = subset[["variable", "n", "median", "IQR", "min", "max"]]
    return subset.to_string(index=False)
