"""
src/id_runner.py
----------------
MODULE 5: Inverse Dynamics via OpenSim Python API.

Reads the IK .mot file (from OpenCap or IK runner), applies optional low-pass
Butterworth filtering to reduce differentiation noise, then runs OpenSim's
InverseDynamicsTool to produce net joint moments.

Key fixes vs original:
  - clip_start_time: 0.0 — clips the slight negative-time offset that some
    OpenCap .mot files carry so that t_start passed to ID is always >= 0.
    This prevents a ~55-row overshoot in the ID output vs GUI.
  - Model must be the OpenCap pre-scaled model (use_opencap_scaled_model: true
    in config) so that segment geometry and inertial parameters match the GUI.
    Using a re-scaled model produces ~49% lower lumbar moments.

Requires: opensim conda package (opensim-org channel, Python 3.10)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.utils import read_mot_file, get_analysis_dir

logger = logging.getLogger(__name__)


def apply_lowpass_filter(
    ik_mot_path: Path,
    output_path: Path,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> Path:
    """
    Apply a zero-lag Butterworth low-pass filter to kinematic data before ID.
    Filtering reduces noise amplification during numerical differentiation inside ID.

    The sampling frequency is inferred from the time column of the .mot file.
    Filter is applied column-wise to all joint angle columns (skipping 'time').

    Parameters
    ----------
    ik_mot_path : Path to input .mot file (unfiltered kinematics)
    output_path : Path to save the filtered .mot file
    cutoff_hz   : Low-pass cutoff frequency (6 Hz is appropriate for slow occupational tasks)
    order       : Butterworth filter order (4th order = 24 dB/octave roll-off)

    Returns
    -------
    Path to the filtered .mot file
    """
    df = read_mot_file(ik_mot_path)

    if "time" not in df.columns:
        logger.warning("No 'time' column in .mot — copying unfiltered to %s", output_path)
        shutil.copy2(ik_mot_path, output_path)
        return output_path

    time = df["time"].to_numpy()
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt  # Sampling frequency (Hz)

    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        logger.warning(
            "Cutoff %.1f Hz >= Nyquist %.1f Hz — skipping filter", cutoff_hz, nyq
        )
        shutil.copy2(ik_mot_path, output_path)
        return output_path

    b, a = butter(order, cutoff_hz / nyq, btype="low")

    data_cols = [c for c in df.columns if c != "time"]
    df_filtered = df.copy()
    for col in data_cols:
        df_filtered[col] = filtfilt(b, a, df[col].to_numpy())

    # Write back as a .mot file preserving the original header structure
    _write_mot_file(df_filtered, ik_mot_path, output_path)
    logger.info(
        "Filtered kinematics saved to %s (cutoff=%.1f Hz, order=%d)",
        output_path.name, cutoff_hz, order,
    )
    return output_path


def run_inverse_dynamics(
    participant_id: str,
    scaled_model_path: Path,
    ik_mot_path: Path,
    participant_output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Run OpenSim Inverse Dynamics for one participant.

    Steps:
    1. Optionally low-pass filter the IK kinematics
    2. Configure opensim.InverseDynamicsTool
    3. Execute ID
    4. Return path to inverse_dynamics.sto

    Parameters
    ----------
    participant_id         : e.g. 'SHIVANGI'
    scaled_model_path      : Path to participant-specific scaled .osim
                             IMPORTANT: must be the OpenCap pre-scaled model
                             (use_opencap_scaled_model: true) — a re-scaled model
                             produces systematically lower ID moments (~49% for lumbar).
    ik_mot_path            : Path to IK output .mot (OpenCap or re-run IK)
    participant_output_dir : Root output dir for this participant
    config                 : Global pipeline config dict

    Returns
    -------
    Path to the produced inverse_dynamics.sto
    """
    try:
        import opensim as osim
    except ImportError:
        raise ImportError(
            "OpenSim Python package not found.\n"
            "Install it by running:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate opensim_pipeline\n"
        )

    id_dir = get_analysis_dir(participant_output_dir, "id")
    id_cfg = config.get("inverse_dynamics", {})

    # --- Step 1: Filter kinematics ---
    cutoff_hz = id_cfg.get("lowpass_filter_freq", 6.0)
    filter_order = id_cfg.get("filter_order", 4)
    filtered_mot_path = id_dir / f"{participant_id}_kinematics_filtered.mot"
    apply_lowpass_filter(ik_mot_path, filtered_mot_path, cutoff_hz, filter_order)

    # --- Step 2: Configure ID tool ---
    output_sto = id_dir / f"{participant_id}_inverse_dynamics.sto"

    model = osim.Model(str(scaled_model_path))
    model.initSystem()

    id_tool = osim.InverseDynamicsTool()
    id_tool.setModel(model)
    id_tool.setCoordinatesFileName(str(filtered_mot_path))
    id_tool.setOutputGenForceFileName(str(output_sto))

    # Time range: read from filtered .mot file.
    # FIX: clip_start_time=0.0 prevents negative-time offsets (from OpenCap)
    # that cause ~55 extra rows vs the GUI and slightly distort the RMS.
    mot_df = read_mot_file(filtered_mot_path)
    if "time" in mot_df.columns:
        clip_start = float(id_cfg.get("clip_start_time", 0.0))
        raw_t_start = float(mot_df["time"].iloc[0])
        t_start = max(raw_t_start, clip_start)
        t_end = float(mot_df["time"].iloc[-1])
        id_tool.setStartTime(t_start)
        id_tool.setEndTime(t_end)
        logger.info(
            "[%s] ID time range: %.4f -> %.4f s  (raw mot start was %.6f s)",
            participant_id, t_start, t_end, raw_t_start,
        )

    id_tool.setResultsDir(str(id_dir))

    # --- Step 3: Run ---
    logger.info(
        "[%s] Running Inverse Dynamics (model=%s, filtered at %.1f Hz)...",
        participant_id, scaled_model_path.name, cutoff_hz,
    )
    id_tool.run()

    if not output_sto.exists():
        raise RuntimeError(f"ID did not produce expected output: {output_sto}")

    logger.info("[%s] ID complete -> %s", participant_id, output_sto.name)
    return output_sto


# ─────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────

def _write_mot_file(df: pd.DataFrame, original_path: Path, output_path: Path) -> None:
    """
    Re-write a filtered DataFrame back to .mot format.
    Copies the original header lines and replaces the data block.
    """
    with open(original_path, "r") as fh:
        original_lines = fh.readlines()

    # Find endheader
    header_lines = []
    header_end_idx = None
    for i, line in enumerate(original_lines):
        header_lines.append(line)
        if line.strip().lower() == "endheader":
            header_end_idx = i
            break

    if header_end_idx is None:
        raise ValueError(f"'endheader' not found in original .mot: {original_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.writelines(header_lines)
        # Write the DataFrame (tab-separated, including column names)
        fh.write(df.to_csv(sep="\t", index=False, lineterminator="\n"))
