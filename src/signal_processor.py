"""
src/signal_processor.py
-----------------------
MODULE 7: RMS computation, full-wave rectification, and per-variable signal extraction.

This module has NO OpenSim dependency — it works on pandas DataFrames produced by
utils.read_sto_file() and utils.read_mot_file().  It reads pre-computed outputs from
the OpenSim GUI (IK .mot, ID .sto, SO activation .sto, SO force .sto).

The RMS formula implemented here EXACTLY matches the thesis Excel formula:
    =SQRT(SUMQ(r1:rn) / COUNTA(r1:rn))
which is equivalent to sqrt(sum(x_i²) / n), excluding NaN values.

Outputs BOTH muscle activation RMS (dimensionless 0-1) AND muscle force RMS (N)
for the configured muscle groups.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import read_sto_file, read_mot_file

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Core math — tested via pytest
# ─────────────────────────────────────────────────────────────

def compute_rms(signal: np.ndarray) -> float:
    """
    Compute RMS matching the thesis Excel formula: sqrt(sum(x²) / n).
    NaN values are excluded (matches COUNTA behaviour in Excel).

    Parameters
    ----------
    signal : np.ndarray
        1-D array of signal values.

    Returns
    -------
    float
        RMS value.
    """
    signal_clean = signal[~np.isnan(signal)]
    if len(signal_clean) == 0:
        return float("nan")
    return float(np.sqrt(np.sum(signal_clean ** 2) / len(signal_clean)))


def full_wave_rectify(signal: np.ndarray) -> np.ndarray:
    """
    Full-wave rectification: apply absolute value to all elements.
    Equivalent to the Excel ABS() function used in the thesis.
    """
    return np.abs(signal)


def full_wave_rectify_then_rms(activation_signal: np.ndarray) -> float:
    """Apply full-wave rectification then compute RMS (thesis method for muscle activations)."""
    rectified = full_wave_rectify(activation_signal)
    return compute_rms(rectified)


def compute_resultant_magnitude(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """
    Compute the 3D vector magnitude time-series: sqrt(Fx² + Fy² + Fz²).
    Used for joint reaction forces before RMS extraction.
    """
    return np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)


# ─────────────────────────────────────────────────────────────
#  Extraction functions
# ─────────────────────────────────────────────────────────────

def _extract_group_rms(
    sto_path: str | Path,
    muscle_groups: dict[str, list[str]],
    label: str = "activation",
) -> dict[str, float]:
    """
    Generic: Read an SO .sto file and compute RMS per muscle group.
    Full-wave rectification is applied before RMS (matches thesis method).

    Works for both activation and force files — the columns are identical,
    only the values differ (0-1 vs Newtons).

    Parameters
    ----------
    sto_path      : path to a *_StaticOptimization_*.sto file
    muscle_groups : dict mapping group label -> list of OpenSim muscle column names
    label         : 'activation' or 'force' (for logging)

    Returns
    -------
    dict mapping group label -> RMS float
    """
    df = read_sto_file(sto_path)
    result: dict[str, float] = {}

    for group_label, muscle_names in muscle_groups.items():
        found_cols = [col for col in muscle_names if col in df.columns]
        missing = set(muscle_names) - set(found_cols)
        if missing:
            logger.warning(
                "[%s] Muscle group '%s': %d muscle(s) not found in .sto — %s",
                label, group_label, len(missing), missing,
            )
        if not found_cols:
            logger.error(
                "[%s] No muscles found for group '%s'. Check muscle_group mapping in config.",
                label, group_label,
            )
            result[group_label] = float("nan")
            continue

        # Average across muscles in the group, then rectify + RMS
        group_signal = df[found_cols].mean(axis=1).to_numpy()
        result[group_label] = full_wave_rectify_then_rms(group_signal)
        logger.debug(
            "[%s] Group '%s': RMS=%.4f (from %d muscles)",
            label, group_label, result[group_label], len(found_cols),
        )

    return result


def extract_muscle_rms(
    so_activation_sto: str | Path,
    muscle_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Extract RMS of muscle activations (dimensionless, 0-1) per group."""
    return _extract_group_rms(so_activation_sto, muscle_groups, label="activation")


def extract_muscle_force_rms(
    so_force_sto: str | Path,
    muscle_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Extract RMS of muscle forces (Newtons) per group."""
    return _extract_group_rms(so_force_sto, muscle_groups, label="force")


def extract_moment_rms(
    id_sto: str | Path,
    joint_dof_map: dict[str, str],
) -> dict[str, float]:
    """
    Read an Inverse Dynamics .sto file and compute RMS per joint DOF.

    Parameters
    ----------
    id_sto : path to inverse_dynamics.sto
    joint_dof_map : dict mapping OpenSim coordinate name -> thesis label
                   (from config['joint_dof_map'])

    Returns
    -------
    dict mapping thesis label -> RMS float (e.g. {'lumbar_flexion_Nm': 43.1, ...})
    """
    df = read_sto_file(id_sto)
    result: dict[str, float] = {}

    for opensim_col, thesis_label in joint_dof_map.items():
        # OpenSim ID columns are named like 'lumbar_extension_moment'
        # Try exact match first, then try appending '_moment'
        col = _find_column(df, [opensim_col, opensim_col + "_moment", opensim_col + "_force"])
        if col is None:
            logger.warning("Column '%s' not found in ID output. Available: %s", opensim_col, list(df.columns))
            result[thesis_label] = float("nan")
            continue

        signal = df[col].to_numpy()
        result[thesis_label] = compute_rms(signal)
        logger.debug("Joint DOF '%s' -> '%s': RMS=%.4f", opensim_col, thesis_label, result[thesis_label])

    return result


def extract_ik_rms(
    ik_mot: str | Path,
    joint_angle_map: dict[str, str],
) -> dict[str, float]:
    """
    Read an IK .mot file and compute RMS per joint angle coordinate.

    Parameters
    ----------
    ik_mot : path to the task-trial .mot file
    joint_angle_map : dict mapping OpenSim coordinate name -> thesis label

    Returns
    -------
    dict mapping thesis label -> RMS angle (degrees)
    """
    df = read_mot_file(ik_mot)
    result: dict[str, float] = {}

    for opensim_col, thesis_label in joint_angle_map.items():
        col = _find_column(df, [opensim_col])
        if col is None:
            logger.warning("Column '%s' not found in IK .mot. Available: %s", opensim_col, list(df.columns))
            result[thesis_label] = float("nan")
            continue

        signal = df[col].to_numpy()
        result[thesis_label] = compute_rms(signal)
        logger.debug("Joint angle '%s' -> '%s': RMS=%.2f deg", opensim_col, thesis_label, result[thesis_label])

    return result


def process_participant_outputs(
    participant_id: str,
    ik_mot: Path,
    so_activation_sto: Path,
    so_force_sto: Path,
    id_sto: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Master function for Module 7.  Runs all extractions for one participant and
    returns a single flat dictionary of RMS values ready to be appended to the
    master dataset.

    Extracts:
      - IK joint angle RMS (degrees)
      - ID joint moment RMS (Nm)
      - SO muscle activation RMS (dimensionless 0-1)
      - SO muscle force RMS (Newtons)

    Also generates per-participant muscle activation plots if configured.
    """
    row: dict[str, Any] = {"participant_id": participant_id}
    muscle_groups = config.get("muscle_groups", {})

    # --- Kinematic RMS ---
    logger.info("[%s] Extracting IK joint angle RMS", participant_id)
    ik_rms = extract_ik_rms(ik_mot, config.get("joint_angle_map", {}))
    row.update(ik_rms)

    # --- Joint Moment RMS (Inverse Dynamics) ---
    logger.info("[%s] Extracting ID joint moment RMS", participant_id)
    id_rms = extract_moment_rms(id_sto, config.get("joint_dof_map", {}))
    row.update(id_rms)

    # --- Muscle Activation RMS (Static Optimization) ---
    logger.info("[%s] Extracting SO muscle activation RMS", participant_id)
    so_act_rms = extract_muscle_rms(so_activation_sto, muscle_groups)
    so_act_renamed = {f"{k}_activation": v for k, v in so_act_rms.items()}
    row.update(so_act_renamed)

    # --- Muscle Force RMS (Static Optimization) ---
    logger.info("[%s] Extracting SO muscle force RMS", participant_id)
    so_force_rms = extract_muscle_force_rms(so_force_sto, muscle_groups)
    so_force_renamed = {f"{k}_force_N": v for k, v in so_force_rms.items()}
    row.update(so_force_renamed)

    # --- Plots ---
    if config.get("output", {}).get("save_plots", True):
        _generate_activation_plots(
            participant_id=participant_id,
            so_activation_sto=so_activation_sto,
            config=config,
            output_dir=output_dir / "plots",
        )

    return row


# ─────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────

def _generate_activation_plots(
    participant_id: str,
    so_activation_sto: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate muscle activation time-series plots for selected muscle groups.
    One PNG per muscle group saved to output_dir/plots/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = config.get("output", {}).get("plot_format", "png")
    groups_to_plot = config.get("output", {}).get("plot_muscles", list(config.get("muscle_groups", {}).keys()))
    muscle_groups = config.get("muscle_groups", {})

    df = read_sto_file(so_activation_sto)
    if "time" not in df.columns:
        logger.warning("No 'time' column found in SO activation file — plots may have incorrect x-axis")
        time_col = df.columns[0]
    else:
        time_col = "time"

    time = df[time_col].to_numpy()

    for group_label in groups_to_plot:
        muscle_names = muscle_groups.get(group_label, [])
        found_cols = [c for c in muscle_names if c in df.columns]
        if not found_cols:
            logger.debug("Skipping plot for '%s' — no muscles found in SO output", group_label)
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        for col in found_cols:
            ax.plot(time, df[col].to_numpy(), linewidth=1.2, label=col)

        # Plot the group mean
        group_mean = df[found_cols].mean(axis=1).to_numpy()
        ax.plot(time, group_mean, color="black", linewidth=2.5, linestyle="--", label=f"{group_label} mean")
        ax.fill_between(time, df[found_cols].min(axis=1), df[found_cols].max(axis=1),
                        alpha=0.15, color="steelblue")

        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Muscle Activation (0–1)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{participant_id} — {group_label.replace('_', ' ').title()}", fontsize=13)
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        out_path = output_dir / f"{participant_id}_{group_label}.{fmt}"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("[%s] Saved plot: %s", participant_id, out_path.name)


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
