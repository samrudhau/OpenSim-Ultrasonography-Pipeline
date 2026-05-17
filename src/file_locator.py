"""
src/file_locator.py
-------------------
Smart discovery of pre-computed OpenSim GUI output files.

Handles the inconsistent file naming across patients:
  - Standard OpenSim auto-generated names (e.g. 'inverse_dynamics.sto',
    'LaiUhlrich2022_scaled-scaled_StaticOptimization_force.sto')
  - Manually renamed files (e.g. 'santhosh inverse dynamics.sto',
    'shivil StaticOptimization force.sto')

Each locator function tries multiple glob patterns in order of specificity,
logs the match, and raises a clear error if no file is found.
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  STO / MOT file validation
# ─────────────────────────────────────────────────────────────

def validate_sto_file(filepath: Path, label: str = "STO") -> int:
    """
    Validate an OpenSim .sto/.mot file has correct structure.

    Checks:
    1. File exists and is non-empty
    2. Contains 'endheader' marker
    3. Has at least one data row after the header
    4. First column is 'time'

    Returns the number of data rows.
    Raises ValueError on structural issues.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"{label} file not found: {filepath}")
    if filepath.stat().st_size == 0:
        raise ValueError(f"{label} file is empty: {filepath}")

    with open(filepath, "r") as fh:
        lines = fh.readlines()

    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break

    if header_end is None:
        raise ValueError(f"{label} file missing 'endheader' marker: {filepath}")

    data_lines = [l for l in lines[header_end + 1:] if l.strip()]
    if len(data_lines) < 2:  # need at least column header + 1 data row
        raise ValueError(
            f"{label} file has no data rows after header: {filepath}"
        )

    # Check first column is 'time'
    col_line = data_lines[0].strip()
    first_col = col_line.split()[0].lower() if col_line else ""
    if first_col != "time":
        raise ValueError(
            f"{label} file first column is '{first_col}', expected 'time': {filepath}"
        )

    n_data_rows = len(data_lines) - 1  # subtract column header line
    return n_data_rows


# ─────────────────────────────────────────────────────────────
#  File locators
# ─────────────────────────────────────────────────────────────

def _search_patterns(directory: Path, patterns: list[str], label: str) -> Path:
    """
    Try each glob pattern in order. Return the first match.
    Raises FileNotFoundError if none match.
    """
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            if len(matches) > 1:
                logger.warning(
                    "[%s] Multiple matches for '%s' in %s — using: %s",
                    label, pattern, directory.name, matches[0].name,
                )
            else:
                logger.debug("[%s] Found: %s", label, matches[0].name)
            return matches[0]

    all_files = sorted(f.name for f in directory.iterdir() if f.is_file())
    raise FileNotFoundError(
        f"[{label}] No file found in {directory}\n"
        f"  Tried patterns: {patterns}\n"
        f"  Files present: {all_files}"
    )


def locate_ik_mot(kinematics_dir: Path, participant_id: str) -> Path:
    """Locate the IK / kinematics .mot file."""
    patterns = [
        "usg*.mot",
        "*.mot",
    ]
    path = _search_patterns(kinematics_dir, patterns, f"{participant_id}/IK")
    return path


def locate_inverse_dynamics(model_dir: Path, participant_id: str) -> Path:
    """
    Locate the Inverse Dynamics .sto file.

    Standard name: 'inverse_dynamics.sto'
    Alt names:     '<name> inverse dynamics.sto', '<name>_inverse_dynamics.sto'
    """
    patterns = [
        "inverse_dynamics.sto",
        "*inverse_dynamics*.sto",
        "*inverse dynamics*.sto",
        "*Inverse*Dynamics*.sto",  # case variations
    ]
    path = _search_patterns(model_dir, patterns, f"{participant_id}/ID")
    return path


def locate_so_activation(model_dir: Path, participant_id: str) -> Path:
    """
    Locate the Static Optimization activation .sto file.

    Standard: '*_StaticOptimization_activation.sto'
    """
    patterns = [
        "*StaticOptimization_activation.sto",
        "*static*optimization*activation*.sto",
        "*Static*Optimization*activation*.sto",
    ]
    path = _search_patterns(model_dir, patterns, f"{participant_id}/SO_act")
    return path


def locate_so_force(model_dir: Path, participant_id: str) -> Path:
    """
    Locate the Static Optimization force .sto file.

    Standard: '*_StaticOptimization_force.sto'
    Alt:      '<name> StaticOptimization force.sto',
              '<name> static optimization.sto' (contains forces, confirmed)
    """
    patterns = [
        "*StaticOptimization_force.sto",
        "*StaticOptimization force.sto",
        "*Static Optimization force.sto",
        # Some patients have just 'static optimization.sto' which contains forces
        "*static optimization.sto",
        "*Static*Optimization*.sto",
    ]
    # Need to exclude activation and controls files from the broad patterns
    all_candidates = []
    for pattern in patterns:
        matches = sorted(model_dir.glob(pattern))
        for m in matches:
            name_lower = m.name.lower()
            if "activation" not in name_lower and "controls" not in name_lower:
                all_candidates.append(m)
        if all_candidates:
            break

    if not all_candidates:
        all_files = sorted(f.name for f in model_dir.iterdir() if f.is_file())
        raise FileNotFoundError(
            f"[{participant_id}/SO_force] No SO force file found in {model_dir}\n"
            f"  Tried patterns: {patterns}\n"
            f"  Files present: {all_files}"
        )

    if len(all_candidates) > 1:
        logger.warning(
            "[%s/SO_force] Multiple candidates — using: %s",
            participant_id, all_candidates[0].name,
        )
    else:
        logger.debug("[%s/SO_force] Found: %s", participant_id, all_candidates[0].name)

    return all_candidates[0]


def locate_all_precomputed(
    kinematics_dir: Path,
    model_dir: Path,
    participant_id: str,
    validate: bool = True,
) -> dict[str, Path]:
    """
    Locate all required pre-computed output files for one participant.

    Returns a dict with keys:
      'ik_mot', 'id_sto', 'so_activation_sto', 'so_force_sto'

    If validate=True, each file is checked for structural correctness.
    """
    result = {
        "ik_mot": locate_ik_mot(kinematics_dir, participant_id),
        "id_sto": locate_inverse_dynamics(model_dir, participant_id),
        "so_activation_sto": locate_so_activation(model_dir, participant_id),
        "so_force_sto": locate_so_force(model_dir, participant_id),
    }

    if validate:
        for key, path in result.items():
            label = f"{participant_id}/{key}"
            n_rows = validate_sto_file(path, label)
            logger.info(
                "[%s] Validated %s: %s (%d data rows)",
                participant_id, key, path.name, n_rows,
            )

    return result
