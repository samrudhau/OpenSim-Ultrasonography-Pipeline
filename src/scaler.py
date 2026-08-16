"""
src/scaler.py
-------------
MODULE 2: Model acquisition / scaling.

Supports three modes, selected by config['model']:

  1. use_buet_model: true  (new default)
     → Locates 'Bilateral Upper Extremity Trunk Model.osim' inside the
       participant's OpenSimData/Model/ folder and copies it to the output
       directory. No scaling is performed — the BUET model is used as-is.

  2. use_opencap_scaled_model: true
     → Finds and copies the OpenCap-produced *_scaled.osim model.

  3. Both false
     → Runs opensim.ScaleTool using participant anthropometrics (advanced).

Requires: opensim conda package ONLY in mode 3.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.utils import find_scaled_model, get_analysis_dir, find_task_trc

logger = logging.getLogger(__name__)


def get_or_scale_model(
    participant_id: str,
    participant_info: dict,
    participant_row: dict,
    participant_output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Return the path to the .osim model for this participant.

    Mode selection (priority order):
      1. config['model']['use_buet_model']: true
         → Copies 'Bilateral Upper Extremity Trunk Model.osim' from the
           participant's OpenSimData/Model/ folder. No scaling.
      2. config['model']['use_opencap_scaled_model']: true
         → Copies the OpenCap-produced *_scaled.osim.
      3. Both false
         → Runs opensim.ScaleTool (advanced, requires opensim package).

    Parameters
    ----------
    participant_id         : e.g. 'AKSHITHA'
    participant_info       : From utils.discover_patient_folders()
    participant_row        : Row from participants.csv (as dict)
    participant_output_dir : Root output dir for this participant
    config                 : Global pipeline config dict

    Returns
    -------
    Path to the .osim model ready for analysis
    """
    model_cfg = config.get("model", {})
    use_buet = model_cfg.get("use_buet_model", False)
    use_opencap = model_cfg.get("use_opencap_scaled_model", True)
    scaled_model_dir = get_analysis_dir(participant_output_dir, "scaled_model")

    if use_buet:
        return _use_buet_model(
            participant_id,
            participant_info["model_dir"],
            scaled_model_dir,
            model_cfg.get("buet_model_name", "Bilateral Upper Extremity Trunk Model.osim"),
            config,
        )
    elif use_opencap:
        return _use_opencap_model(
            participant_id,
            participant_info["model_dir"],
            scaled_model_dir,
            model_cfg.get("scaled_model_pattern", "*_scaled.osim"),
            config,
        )
    else:
        return _run_opensim_scaler(
            participant_id,
            participant_info,
            participant_row,
            scaled_model_dir,
            config,
        )


def _use_buet_model(
    participant_id: str,
    model_dir: Path,
    output_dir: Path,
    buet_model_name: str,
    config: dict[str, Any],
) -> Path:
    """
    Locate the BUET model by filename in the participant's Model/ directory
    and copy it to the pipeline output directory. No scaling is applied.
    """
    source = model_dir / buet_model_name
    if not source.exists():
        raise FileNotFoundError(
            f"BUET model '{buet_model_name}' not found in {model_dir}\n"
            f"Files present: {[f.name for f in model_dir.iterdir() if f.suffix == '.osim']}"
        )

    dest = output_dir / f"{participant_id}_buet_model.osim"
    overwrite = config.get("output", {}).get("overwrite_existing", False)

    if not dest.exists() or overwrite:
        shutil.copy2(source, dest)
        logger.info(
            "[%s] Copied BUET model (no scaling): %s → %s",
            participant_id, source.name, dest.name,
        )
    else:
        logger.info(
            "[%s] BUET model already copied at %s — skipping.",
            participant_id, dest.name,
        )

    return dest


def _use_opencap_model(
    participant_id: str,
    model_dir: Path,
    output_dir: Path,
    pattern: str,
    config: dict[str, Any],
) -> Path:
    """Locate the OpenCap-scaled model and copy it to the output directory."""
    source = find_scaled_model(model_dir, pattern)
    dest = output_dir / f"{participant_id}_scaled.osim"

    overwrite = config.get("output", {}).get("overwrite_existing", False)

    if not dest.exists() or overwrite:
        shutil.copy2(source, dest)
        logger.info("[%s] Copied OpenCap-scaled model: %s → %s", participant_id, source.name, dest.name)
    else:
        logger.info("[%s] Scaled model already exists at %s — skipping copy", participant_id, dest.name)

    return dest


def _run_opensim_scaler(
    participant_id: str,
    participant_info: dict,
    participant_row: dict,
    scaled_model_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Run opensim.ScaleTool to re-scale the OpenCap model using neutral.trc for
    measurements and usg.trc for static pose adjustments to match the GUI steps.
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

    # 1. Base Model -> The OpenCap scaled model
    model_cfg = config.get("model", {})
    base_model_path = find_scaled_model(
        participant_info["model_dir"],
        model_cfg.get("scaled_model_pattern", "*_scaled.osim")
    )

    # 2. Marker Data Files
    neutral_trc = participant_info["marker_data_dir"] / next(
        participant_info["marker_data_dir"].glob(
            config.get("neutral_trc_pattern", "neutral*.trc")
        )
    )
    
    task_trc = find_task_trc(
        participant_info["marker_data_dir"],
        config.get("task_trc_pattern", "usg*.trc")
    )

    output_model_path = scaled_model_dir / f"{participant_id}_scaled.osim"
    
    overwrite = config.get("output", {}).get("overwrite_existing", False)
    if output_model_path.exists() and not overwrite:
        logger.info("[%s] Scaled model already exists — skipping re-scale", participant_id)
        return output_model_path

    # Build scale tool
    scale_tool = osim.ScaleTool()
    scale_tool.setName(participant_id)
    scale_tool.setSubjectMass(float(participant_row.get("weight_kg", 70.0)))
    scale_tool.setSubjectHeight(float(participant_row.get("height_m", 1.70)) * 100)  # cm

    # Generic Model Maker
    generic_model_maker = scale_tool.getGenericModelMaker()
    generic_model_maker.setModelFileName(str(base_model_path))
    generic_model_maker.setMarkerSetFileName(str(task_trc))

    # Model Scaler
    model_scaler = scale_tool.getModelScaler()
    model_scaler.setMarkerFileName(str(task_trc))
    
    # Set ModelScaler time range
    neutral_marker_data = osim.MarkerData(str(task_trc))
    neutral_time_range = osim.ArrayDouble()
    neutral_time_range.append(neutral_marker_data.getStartFrameTime())
    neutral_time_range.append(neutral_marker_data.getLastFrameTime())
    model_scaler.setTimeRange(neutral_time_range)
    
    # Marker Placer
    marker_placer = scale_tool.getMarkerPlacer()
    marker_placer.setStaticPoseFileName(str(neutral_trc))
    
    # Set MarkerPlacer time range
    task_marker_data = osim.MarkerData(str(task_trc))
    task_time_range = osim.ArrayDouble()
    task_time_range.append(task_marker_data.getStartFrameTime())
    task_time_range.append(task_marker_data.getLastFrameTime())
    marker_placer.setTimeRange(task_time_range)
    
    marker_placer.setOutputModelFileName(str(output_model_path))

    logger.info("[%s] Running ScaleTool (base=%s, scale_marker=%s, pose_marker=%s)...", 
                participant_id, base_model_path.name, neutral_trc.name, task_trc.name)
    scale_tool.run()

    if not output_model_path.exists():
        raise RuntimeError(f"ScaleTool did not produce expected output: {output_model_path}")

    logger.info("[%s] Scaling complete → %s", participant_id, output_model_path.name)
    return output_model_path
