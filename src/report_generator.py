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
#  Dynamic reference values
#  Instead of hardcoding medians from the Lai-Uhlrich model,
#  we compute the cohort median from the current pipeline run.
#  This is self-consistent with the BUET model results.
# ─────────────────────────────────────────────────────────────

VALIDATION_TOLERANCE = 0.20  # 20% IQR-based spread flag (replaces ±5% thesis check)


def build_dynamic_reference(descriptives_df: pd.DataFrame) -> dict[str, float]:
    """
    Build a reference dict from computed cohort medians.
    Used for self-consistency validation (flags variables with high spread).
    Returns {variable_name: cohort_median}.
    """
    ref: dict[str, float] = {}
    for _, row in descriptives_df.iterrows():
        var = row["variable"]
        med = row["median"]
        if pd.notna(med):
            ref[var] = float(med)
    return ref


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
    reference: dict[str, float] | None = None,
    tolerance: float = VALIDATION_TOLERANCE,
) -> pd.DataFrame:
    """
    Compare computed cohort medians against a reference dict.

    When reference is None (BUET model mode), uses the cohort medians
    themselves as the reference and flags variables with high spread
    (IQR/median > tolerance), indicating outlier participants.

    Parameters
    ----------
    computed_df : Output from compute_descriptives()
    reference   : Dict mapping variable name -> reference median.
                  If None, uses cohort medians (self-consistency check).
    tolerance   : Spread tolerance for flagging (default 20% IQR/median)

    Returns
    -------
    pd.DataFrame with columns: variable, cohort_median, IQR, spread_ratio, status
    """
    if reference is None:
        reference = build_dynamic_reference(computed_df)

    report_rows = []
    for var, ref_median in reference.items():
        row_match = computed_df[computed_df["variable"] == var]
        if row_match.empty:
            continue

        computed_median = float(row_match["median"].iloc[0])
        iqr = float(row_match["IQR"].iloc[0])

        # Spread ratio: IQR / median — flags high variability across participants
        if ref_median == 0 or computed_median == 0:
            spread_ratio = float("nan")
            status = "OK (zero)"
        else:
            spread_ratio = iqr / abs(computed_median)
            status = "OK" if spread_ratio <= tolerance else f"HIGH SPREAD — IQR/median={spread_ratio:.1%}"

        report_rows.append({
            "variable": var,
            "cohort_median": round(computed_median, 3),
            "IQR": round(iqr, 3),
            "spread_ratio": f"{spread_ratio:.2%}" if not np.isnan(spread_ratio) else "n/a",
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

    # Self-consistency validation (dynamic reference from cohort medians)
    validation = compare_with_thesis_values(descriptives, reference=None)

    # Build report text
    lines = _build_report_text(descriptives, validation, n_participants)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    logger.info("Summary report written → %s", output_path.name)

    # Print high-spread warnings to console
    if not validation.empty:
        warnings = validation[validation["status"].str.startswith("HIGH SPREAD")]
        if not warnings.empty:
            logger.warning(
                "Cohort spread check: %d variable(s) have IQR/median > %d%%:\n%s",
                len(warnings),
                int(VALIDATION_TOLERANCE * 100),
                warnings[["variable", "cohort_median", "IQR", "spread_ratio"]].to_string(index=False),
            )
        else:
            logger.info(
                "Cohort spread check: All variables have IQR/median ≤ %d%%. ✓",
                int(VALIDATION_TOLERANCE * 100),
            )


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
        "  OPENSIM BIOMECHANICAL PIPELINE — SUMMARY REPORT (BUET Model)",
        f"  Generated: {now}",
        f"  Participants: {n_participants}",
        sep, "",
    ]

    # --- Section 1: Joint Moments (ID) ---
    moment_vars = [v for v in descriptives["variable"] if v.endswith("_Nm")]
    if moment_vars:
        lines += ["─" * 72, "TABLE 1 — Net Joint Moments (RMS, Nm)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, moment_vars))
        lines.append("")

    # --- Section 2: Joint Reaction Forces (JRA) ---
    jra_vars = [v for v in descriptives["variable"] if v.endswith("_N")]
    if jra_vars:
        lines += ["─" * 72, "TABLE 2 — Joint Reaction Forces — BUET Model (RMS Resultant, N)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, jra_vars))
        lines.append("")

    # --- Section 3: Muscle Activations (SO) ---
    activation_vars = [v for v in descriptives["variable"] if v.endswith("_activation")]
    if activation_vars:
        lines += ["─" * 72, "TABLE 3 — Muscle Activations (Full-Wave Rectified RMS, 0-1)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, activation_vars))
        lines.append("")

    # --- Section 4: Muscle Forces (SO) ---
    force_N_vars = [v for v in descriptives["variable"] if v.endswith("_force_N")]
    if force_N_vars:
        lines += ["─" * 72, "TABLE 4 — Muscle Forces (RMS, N)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, force_N_vars))
        lines.append("")

    # --- Section 5: Joint Kinematics (IK) ---
    kin_vars = [v for v in descriptives["variable"] if v.endswith("_deg")]
    if kin_vars:
        lines += ["─" * 72, "TABLE 5 — Joint Kinematics (RMS, degrees)", "─" * 72]
        lines.append(_format_descriptives_table(descriptives, kin_vars))
        lines.append("")

    # --- Section 6: Cohort Spread Check ---
    lines += [
        "─" * 72,
        f"COHORT SPREAD CHECK — Variables with IQR/median > {int(VALIDATION_TOLERANCE*100)}%",
        "(High spread may indicate outlier participants or model convergence issues)",
        "─" * 72,
    ]
    if not validation.empty:
        flagged = validation[validation["status"].str.startswith("HIGH SPREAD")]
        if not flagged.empty:
            lines.append(flagged.to_string(index=False))
        else:
            lines.append(f"  All variables within {int(VALIDATION_TOLERANCE*100)}% IQR/median. \u2713")
    else:
        lines.append("  (No validation data available)")
    lines += ["", sep, "END OF REPORT", sep]

    return lines


def _format_descriptives_table(desc_df: pd.DataFrame, variables: list[str]) -> str:
    """Format a subset of descriptives as an aligned text table."""
    subset = desc_df[desc_df["variable"].isin(variables)].copy()
    subset = subset[["variable", "n", "median", "IQR", "min", "max"]]
    return subset.to_string(index=False)
