"""
src/ik_runner.py
----------------
MODULE 3: Inverse Kinematics via OpenSim Python API.

By default (config['inverse_kinematics']['use_opencap_kinematics'] = true),
this module SKIPS re-running IK and simply returns the path to the existing
OpenCap .mot file — OpenCap already runs IK during data processing.

Set use_opencap_kinematics: false in pipeline_config.yaml to force re-running
IK from the .trc files (useful if you want to apply different IK settings).

Requires: opensim conda package ONLY if use_opencap_kinematics=False
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils import (
    find_task_mot,
    find_task_trc,
    find_neutral_trc,
    get_analysis_dir,
)

logger = logging.getLogger(__name__)


def run_inverse_kinematics(
    participant_id: str,
    participant_info: dict,
    scaled_model_path: Path,
    participant_output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Return the path to the IK output .mot file for this participant.

    If config['inverse_kinematics']['use_opencap_kinematics'] is True:
      → Returns path to the existing OpenCap .mot in OpenSimData/Kinematics/

    If False:
      → Runs opensim.InverseKinematicsTool and returns the new .mot

    Parameters
    ----------
    participant_id         : e.g. 'SHIVANGI'
    participant_info       : From utils.discover_patient_folders()
    scaled_model_path      : Path to the scaled .osim model
    participant_output_dir : Root output dir for this participant
    config                 : Global pipeline config dict

    Returns
    -------
    Path to the .mot file containing joint angle kinematics
    """
    ik_cfg = config.get("inverse_kinematics", {})
    use_opencap = ik_cfg.get("use_opencap_kinematics", True)

    if use_opencap:
        mot_path = find_task_mot(
            participant_info["kinematics_dir"],
            config.get("task_mot_pattern", "*.mot"),
        )
        logger.info("[%s] Using OpenCap kinematics: %s", participant_id, mot_path.name)
        return mot_path
    else:
        return _run_opensim_ik(
            participant_id,
            participant_info,
            scaled_model_path,
            participant_output_dir,
            ik_cfg,
            config,
        )


def _run_opensim_ik(
    participant_id: str,
    participant_info: dict,
    scaled_model_path: Path,
    participant_output_dir: Path,
    ik_cfg: dict,
    config: dict[str, Any],
) -> Path:
    """Run OpenSim InverseKinematicsTool from .trc files."""
    try:
        import opensim as osim
    except ImportError:
        raise ImportError(
            "OpenSim Python package not found.\n"
            "Install it by running:\n"
            "  conda env create -f environment.yml\n"
            "  conda activate opensim_pipeline\n"
        )

    ik_dir = get_analysis_dir(participant_output_dir, "ik")
    output_mot = ik_dir / f"{participant_id}_ik_output.mot"

    if output_mot.exists() and not config.get("output", {}).get("overwrite_existing", False):
        logger.info("[%s] IK output already exists — skipping re-run", participant_id)
        return output_mot

    task_trc = find_task_trc(
        participant_info["marker_data_dir"],
        config.get("task_trc_pattern", "usg*.trc"),
    )

    model = osim.Model(str(scaled_model_path))
    model.initSystem()

    ik_tool = osim.InverseKinematicsTool()
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(str(task_trc))
    ik_tool.setOutputMotionFileName(str(output_mot))
    try:
        ik_tool.set_accuracy(ik_cfg.get("ik_accuracy", 1.0e-8))
    except AttributeError:
        try:
            ik_tool.setAccuracy(ik_cfg.get("ik_accuracy", 1.0e-8))
        except AttributeError:
            pass  # Fall back to default if property is inaccessible

    logger.info("[%s] Running Inverse Kinematics on %s...", participant_id, task_trc.name)
    ik_tool.run()

    if not output_mot.exists():
        raise RuntimeError(f"IK did not produce expected output: {output_mot}")

    # Quality check
    _check_ik_residuals(ik_dir, participant_id, threshold=ik_cfg.get("rms_residual_threshold_cm", 2.0))
    logger.info("[%s] IK complete → %s", participant_id, output_mot.name)
    return output_mot


def _check_ik_residuals(ik_dir: Path, participant_id: str, threshold: float = 2.0) -> None:
    """
    Parse opensim.log to check IK marker RMS residuals.
    Warns if any marker exceeds the threshold (default 2 cm).
    """
    log_path = ik_dir / "opensim.log"
    if not log_path.exists():
        logger.debug("opensim.log not found in IK dir — skipping residual check")
        return

    with open(log_path, "r") as fh:
        log_content = fh.read()

    import re
    # OpenSim reports: "Marker <name> ... RMS = X cm"
    rms_matches = re.findall(r"(\S+)\s+.*?RMS\s*=\s*([\d.]+)\s*cm", log_content)
    for marker_name, rms_str in rms_matches:
        rms_val = float(rms_str)
        if rms_val > threshold:
            logger.warning(
                "[%s] IK marker '%s' RMS residual = %.2f cm > threshold %.2f cm. "
                "Check scaling or marker placement.",
                participant_id, marker_name, rms_val, threshold
            )
