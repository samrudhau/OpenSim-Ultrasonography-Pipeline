"""
src/mot_adapter.py
------------------
MODULE: .mot coordinate adapter for BUET model compatibility.

The OpenCap pipeline produces kinematics using the Lai-Uhlrich coordinate names,
but the Bilateral Upper Extremity Trunk (BUET) model uses different names for the
three lumbar DOFs:

  OpenCap / Lai-Uhlrich   →   BUET model
  ─────────────────────────────────────────
  lumbar_extension         →   flex_extension
  lumbar_bending           →   lat_bending
  lumbar_rotation          →   axial_rotation

This module reads the raw OpenCap .mot file, renames those three columns in memory,
and writes an adapted .mot to the participant's output directory.

The adapted file is used as input to ID, SO, and JRA runners.
The original .mot file on disk is never modified.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.utils import read_mot_file

logger = logging.getLogger(__name__)

# Default coordinate rename map (OpenCap → BUET model)
_DEFAULT_RENAME_MAP: dict[str, str] = {
    "lumbar_extension": "flex_extension",
    "lumbar_bending":   "lat_bending",
    "lumbar_rotation":  "axial_rotation",
}


def adapt_mot_for_buet(
    ik_mot_path: Path,
    output_dir: Path,
    participant_id: str,
    config: dict[str, Any],
) -> Path:
    """
    Rename lumbar coordinate columns in a .mot file to match the BUET model.

    Reads the rename map from config['lumbar_coord_rename'] if present,
    otherwise uses the default OpenCap→BUET mapping above.

    Parameters
    ----------
    ik_mot_path    : Path to the original OpenCap .mot file
    output_dir     : Directory where the adapted .mot will be saved
    participant_id : e.g. 'AKSHITHA'
    config         : Global pipeline config dict

    Returns
    -------
    Path to the adapted .mot file (ready for ID/SO/JRA)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = output_dir / f"{participant_id}_buet_adapted.mot"

    overwrite = config.get("output", {}).get("overwrite_existing", False)
    if adapted_path.exists() and not overwrite:
        logger.info(
            "[%s] Adapted .mot already exists — skipping. Set overwrite_existing: true to re-run.",
            participant_id,
        )
        return adapted_path

    rename_map: dict[str, str] = config.get("lumbar_coord_rename", _DEFAULT_RENAME_MAP)

    # --- Read the original file preserving the header ---
    with open(ik_mot_path, "r", encoding="utf-8") as fh:
        original_lines = fh.readlines()

    # Locate endheader
    header_lines: list[str] = []
    header_end_idx: int | None = None
    for i, line in enumerate(original_lines):
        header_lines.append(line)
        if line.strip().lower() == "endheader":
            header_end_idx = i
            break

    if header_end_idx is None:
        logger.warning(
            "[%s] 'endheader' not found in %s — copying file unchanged",
            participant_id, ik_mot_path.name,
        )
        shutil.copy2(ik_mot_path, adapted_path)
        return adapted_path

    data_lines = original_lines[header_end_idx + 1:]
    if not data_lines:
        logger.warning("[%s] No data lines after endheader — copying file unchanged", participant_id)
        shutil.copy2(ik_mot_path, adapted_path)
        return adapted_path

    # --- Rename columns in the column-header row (first data line) ---
    col_header_line = data_lines[0]
    renamed_cols: list[str] = []
    applied: list[str] = []

    for col in col_header_line.rstrip("\r\n").split("\t"):
        if col in rename_map:
            renamed_cols.append(rename_map[col])
            applied.append(f"  {col} → {rename_map[col]}")
        else:
            renamed_cols.append(col)

    if applied:
        logger.info(
            "[%s] Renaming %d lumbar coordinate(s) for BUET model compatibility:\n%s",
            participant_id, len(applied), "\n".join(applied),
        )
    else:
        logger.info(
            "[%s] No lumbar coordinates required renaming (file may already be BUET-compatible).",
            participant_id,
        )

    new_col_header = "\t".join(renamed_cols) + "\n"

    # --- Write adapted .mot ---
    with open(adapted_path, "w", encoding="utf-8") as fh:
        fh.writelines(header_lines)
        fh.write(new_col_header)
        fh.writelines(data_lines[1:])

    logger.info("[%s] Adapted .mot written → %s", participant_id, adapted_path.name)
    return adapted_path
