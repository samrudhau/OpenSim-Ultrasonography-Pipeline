"""
src/dataset_compiler.py
-----------------------
MODULE 8: Master dataset assembly.

Collects per-participant RMS scalar dictionaries, merges with demographic and
anthropometric data from participants.csv, and writes a single wide-format CSV
that is directly importable into JAMOVI, SPSS, or R.

No OpenSim dependency — pure pandas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Column schema (matches thesis Tables 1-3 variable names)
# ─────────────────────────────────────────────────────────────

# Order in which columns appear in the master CSV
COLUMN_ORDER = [
    # Demographics
    "participant_id", "age", "sex", "dominant_hand", "years_experience",
    # Workload
    "work_hrs_per_week", "scanning_hrs_per_week", "patients_per_day",
    "min_per_patient", "probe_weight_g", "chair_height_cm",
    # Anthropometrics
    "height_m", "weight_kg", "sitting_height_cm", "sitting_shoulder_height_cm",
    "sitting_elbow_height_cm", "sitting_eye_height_cm", "thigh_thickness_cm",
    "buttock_knee_cm", "buttock_popliteal_cm", "knee_height_cm",
    "popliteal_height_cm", "shoulder_breadth_cm", "shoulder_elbow_cm",
    "elbow_fingertip_cm", "shoulder_grip_cm",
    # IK joint angles (RMS, degrees)
    "lumbar_flexion_deg", "lumbar_sideflex_deg", "lumbar_rotation_deg",
    "shoulder_flexion_deg", "shoulder_abduction_deg", "shoulder_rotation_deg",
    "elbow_flexion_deg", "forearm_pronation_deg",
    # Joint moments from ID (RMS, Nm)
    "lumbar_flexion_Nm", "lumbar_sideflex_Nm", "lumbar_rotation_Nm",
    "arm_flexion_Nm", "arm_abduction_Nm", "arm_rotation_Nm",
    "elbow_flexion_Nm", "prosup_Nm",
    # Muscle activations from SO (RMS, dimensionless 0-1)
    "hip_flexors_activation", "hip_extensors_activation",
    "hip_abductors_activation", "hip_adductors_activation",
    "knee_extensors_activation", "knee_flexors_activation",
    "ankle_plantarflexors_activation", "ankle_dorsiflexors_activation",
    # Muscle forces from SO (RMS, Newtons)
    "hip_flexors_force_N", "hip_extensors_force_N",
    "hip_abductors_force_N", "hip_adductors_force_N",
    "knee_extensors_force_N", "knee_flexors_force_N",
    "ankle_plantarflexors_force_N", "ankle_dorsiflexors_force_N",
]


def compile_master_dataset(
    results: list[dict[str, Any]],
    participants_csv_path: str | Path,
    output_path: str | Path,
    jamovi_output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Merge per-participant RMS result rows with anthropometric data and write CSV.

    Parameters
    ----------
    results               : List of dicts, one per participant, from signal_processor
    participants_csv_path : Path to config/participants.csv
    output_path           : Where to write results/master_dataset.csv
    jamovi_output_path    : Optional — write a JAMOVI-compatible version too

    Returns
    -------
    pd.DataFrame  (the master dataset)
    """
    if not results:
        logger.warning("No participant results to compile. Master dataset will be empty.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    logger.info("Compiled results for %d participant(s)", len(results_df))

    # Load anthropometric data
    anthro_df = load_participants_csv(participants_csv_path)

    # Merge on participant_id
    master_df = _merge_with_anthropometrics(results_df, anthro_df)

    # Reorder columns to match thesis schema (missing cols will be at the end)
    ordered_cols = [c for c in COLUMN_ORDER if c in master_df.columns]
    remaining_cols = [c for c in master_df.columns if c not in ordered_cols]
    master_df = master_df[ordered_cols + remaining_cols]

    # Write master CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info("Master dataset written → %s (%d rows × %d cols)",
                output_path.name, len(master_df), len(master_df.columns))

    # Write JAMOVI-compatible version
    if jamovi_output_path:
        write_jamovi_csv(master_df, jamovi_output_path)

    return master_df


def load_participants_csv(participants_csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the participants.csv file."""
    participants_csv_path = Path(participants_csv_path)
    if not participants_csv_path.exists():
        raise FileNotFoundError(f"participants.csv not found: {participants_csv_path}")
    df = pd.read_csv(participants_csv_path, dtype={"participant_id": str})
    df.columns = df.columns.str.strip()
    logger.info("Loaded %d participants from %s", len(df), participants_csv_path.name)
    return df


def _merge_with_anthropometrics(results_df: pd.DataFrame, anthro_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join results with anthropometric data on participant_id."""
    merged = results_df.merge(anthro_df, on="participant_id", how="left", suffixes=("", "_anthro"))
    # Drop duplicate columns from merge
    dup_cols = [c for c in merged.columns if c.endswith("_anthro")]
    merged.drop(columns=dup_cols, inplace=True)
    return merged


def write_jamovi_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Write a JAMOVI-compatible CSV.
    JAMOVI requirements: UTF-8, column names ≤ 64 chars, no special characters in names.
    """
    output_path = Path(output_path)
    jamovi_df = df.copy()
    # Sanitise column names for JAMOVI (replace / and spaces)
    jamovi_df.columns = [
        col.replace("/", "_per_").replace(" ", "_").replace("(", "").replace(")", "")
        for col in jamovi_df.columns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jamovi_df.to_csv(output_path, index=False, float_format="%.4f", encoding="utf-8-sig")
    logger.info("JAMOVI-compatible CSV written → %s", output_path.name)


def collect_participant_results(
    participant_ids: list[str],
    outputs_dir: str | Path,
) -> list[dict[str, Any]]:
    """
    Load previously-processed per-participant result CSV files (if the pipeline
    saves intermediate results) and return them as a list of dicts.
    Used when re-running the compilation step without re-running analysis.
    """
    outputs_dir = Path(outputs_dir)
    rows = []
    for pid in participant_ids:
        result_csv = outputs_dir / pid / f"{pid}_results.csv"
        if result_csv.exists():
            df = pd.read_csv(result_csv)
            rows.append(df.iloc[0].to_dict())
            logger.debug("Loaded cached results for %s", pid)
        else:
            logger.warning("No cached results for %s at %s", pid, result_csv)
    return rows


def save_participant_result(row: dict[str, Any], participant_output_dir: Path, participant_id: str) -> None:
    """Save a single participant's result row as a CSV for caching."""
    result_csv = participant_output_dir / f"{participant_id}_results.csv"
    pd.DataFrame([row]).to_csv(result_csv, index=False, float_format="%.4f")
    logger.debug("Cached results for %s → %s", participant_id, result_csv.name)
