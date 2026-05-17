"""
src/utils.py
------------
Shared utility functions used across pipeline modules.
Includes: .sto/.mot file parsers, patient folder discovery, config loader.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Config loading
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/pipeline_config.yaml") -> dict[str, Any]:
    """Load the global YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    logger.debug("Loaded config from %s", config_path)
    return config


# ─────────────────────────────────────────────────────────────
#  OpenSim file parsers
# ─────────────────────────────────────────────────────────────

def read_sto_file(filepath: str | Path) -> pd.DataFrame:
    """
    Parse an OpenSim .sto file into a pandas DataFrame.

    OpenSim .sto files have a plain-text header section that ends with the
    line 'endheader', followed by a tab-separated data block (column names
    on the first data row, then numerical values).

    Parameters
    ----------
    filepath : str or Path
        Path to the .sto file.

    Returns
    -------
    pd.DataFrame
        DataFrame with time as the first column and one column per variable.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f".sto file not found: {filepath}")

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # Find the 'endheader' marker
    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break

    if header_end is None:
        raise ValueError(f"'endheader' not found in {filepath}. File may be corrupt.")

    # The line immediately after endheader is the column names
    data_lines = lines[header_end + 1:]
    if not data_lines:
        raise ValueError(f"No data found after 'endheader' in {filepath}")

    # Parse as tab-separated (OpenSim default)
    from io import StringIO
    data_str = "".join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep=r"\s+", engine="python")
    logger.debug("Parsed %s → shape %s", filepath.name, df.shape)
    return df


def read_mot_file(filepath: str | Path) -> pd.DataFrame:
    """
    Parse an OpenSim .mot file into a DataFrame.
    .mot files share the same header format as .sto files.
    """
    return read_sto_file(filepath)  # Identical format


# ─────────────────────────────────────────────────────────────
#  Patient folder discovery
# ─────────────────────────────────────────────────────────────

def discover_patient_folders(data_root: str | Path, config: dict) -> list[dict]:
    """
    Scan *data_root* for subdirectories that look like OpenCap session folders.
    A folder qualifies if it contains all sub-directories listed in
    config['patient_folder_markers'] (default: 'MarkerData' and 'OpenSimData').

    Returns a list of dicts, each with:
      - 'folder_name': the directory name (e.g. 'SHIVANGI overall')
      - 'folder_path': absolute Path to the folder
      - 'participant_id': sanitised ID (e.g. 'SHIVANGI')
      - 'marker_data_dir': Path to MarkerData/
      - 'opensim_data_dir': Path to OpenSimData/
      - 'kinematics_dir': Path to OpenSimData/Kinematics/
      - 'model_dir': Path to OpenSimData/Model/
    """
    data_root = Path(data_root)
    markers = config.get("patient_folder_markers", ["MarkerData", "OpenSimData"])

    patients: list[dict] = []
    for folder in sorted(data_root.iterdir()):
        if not folder.is_dir():
            continue
        if all((folder / m).is_dir() for m in markers):
            pid = _sanitise_participant_id(folder.name)
            patients.append({
                "folder_name": folder.name,
                "folder_path": folder,
                "participant_id": pid,
                "marker_data_dir": folder / "MarkerData",
                "opensim_data_dir": folder / "OpenSimData",
                "kinematics_dir": folder / "OpenSimData" / "Kinematics",
                "model_dir": folder / "OpenSimData" / "Model",
            })
            logger.info("Discovered patient: %s → %s", pid, folder)

    if not patients:
        logger.warning("No patient folders found in %s", data_root)
    return patients


def _sanitise_participant_id(folder_name: str) -> str:
    """Turn 'SHIVANGI overall' → 'SHIVANGI'."""
    # Remove common suffixes, strip spaces, make uppercase
    clean = re.sub(r"\s+(overall|data|session)$", "", folder_name, flags=re.IGNORECASE)
    clean = re.sub(r"[^A-Za-z0-9_]", "_", clean).strip("_")
    return clean.upper()


# ─────────────────────────────────────────────────────────────
#  File finders
# ─────────────────────────────────────────────────────────────

def find_file(directory: str | Path, pattern: str, label: str = "file") -> Path:
    """
    Find the first file matching *pattern* (glob) in *directory*.
    Raises FileNotFoundError if no match.
    """
    directory = Path(directory)
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No {label} matching '{pattern}' found in {directory}\n"
            f"Files present: {[f.name for f in directory.iterdir() if f.is_file()]}"
        )
    if len(matches) > 1:
        logger.warning("Multiple %s matches for '%s' in %s — using first: %s",
                       label, pattern, directory, matches[0].name)
    return matches[0]


def find_scaled_model(model_dir: Path, pattern: str = "*_scaled_scaled.osim") -> Path:
    """Locate the OpenCap-scaled .osim model inside a patient's Model/ directory."""
    return find_file(model_dir, pattern, label="scaled .osim model")


def find_task_mot(kinematics_dir: Path, pattern: str = "*.mot") -> Path:
    """Locate the task-trial .mot kinematic file."""
    return find_file(kinematics_dir, pattern, label="task .mot file")


def find_neutral_trc(marker_data_dir: Path, pattern: str = "neutral*.trc") -> Path:
    """Locate the neutral (static) .trc marker file."""
    return find_file(marker_data_dir, pattern, label="neutral .trc file")


def find_task_trc(marker_data_dir: Path, pattern: str = "usg*.trc") -> Path:
    """Locate the task-trial .trc marker file."""
    return find_file(marker_data_dir, pattern, label="task .trc file")


# ─────────────────────────────────────────────────────────────
#  Output directory helpers
# ─────────────────────────────────────────────────────────────

def get_participant_output_dir(output_root: str | Path, participant_id: str) -> Path:
    """Return and create the per-participant output directory."""
    out = Path(output_root) / participant_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_analysis_dir(participant_output_dir: Path, analysis: str) -> Path:
    """Return and create a sub-directory for a specific analysis step."""
    d = participant_output_dir / analysis
    d.mkdir(parents=True, exist_ok=True)
    return d
