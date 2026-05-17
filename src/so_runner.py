"""
src/so_runner.py
----------------
MODULE 4: Static Optimization via OpenSim Python API.

Estimates individual muscle force contributions by minimising the sum of squared
muscle activations subject to joint moment constraints.

Strategy: The programmatic AnalyzeTool API has an internal disconnect between
the model that runs the optimisation and the one whose printResults is called,
so results are never written.  The GUI avoids this by using XML setup files.
This module mirrors that workflow:
  1. Save the model (with reserve actuators) to a temp .osim file.
  2. Write an AnalyzeTool setup XML referencing that model + SO analysis.
  3. Load the AnalyzeTool from the XML and call run().

Key fixes vs original:
  - lock_coordinates now defaults to [] — matching GUI behaviour (no locking).
    Locking pelvis/ankle suppressed muscle forces by 70-82%; GUI uses reserve
    actuators for those DOFs instead.
  - clip_start_time: 0.0 — clips negative-time offsets from OpenCap .mot files
    so the SO time grid starts at 0 and step_interval=4 yields ~240 rows
    (matching the GUI's ~0.0667 s SO step).
  - t_end is taken from the IK .mot file directly (no extra padding).

Outputs:
  - *_StaticOptimization_activation.sto  (dimensionless, 0-1)
  - *_StaticOptimization_force.sto       (Newtons)

Requires: opensim conda package (opensim-org channel, Python 3.10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils import get_analysis_dir, read_mot_file

logger = logging.getLogger(__name__)

_SETUP_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <AnalyzeTool name="{tool_name}">
    <model_file>{model_file}</model_file>
    <results_directory>{results_dir}</results_directory>
    <output_precision>8</output_precision>
    <initial_time>{t_start}</initial_time>
    <final_time>{t_end}</final_time>
    <solve_for_equilibrium_for_auxiliary_states>false</solve_for_equilibrium_for_auxiliary_states>
    <maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>
    <maximum_integrator_step_size>1</maximum_integrator_step_size>
    <minimum_integrator_step_size>1e-08</minimum_integrator_step_size>
    <integrator_error_tolerance>1e-05</integrator_error_tolerance>
    <AnalysisSet>
      <objects>
        <StaticOptimization name="StaticOptimization">
          <on>true</on>
          <start_time>{t_start}</start_time>
          <end_time>{t_end}</end_time>
          <step_interval>{step_interval}</step_interval>
          <in_degrees>true</in_degrees>
          <activation_exponent>{activation_exponent}</activation_exponent>
          <use_model_force_set>{use_model_force_set}</use_model_force_set>
        </StaticOptimization>
      </objects>
    </AnalysisSet>
    <coordinates_file>{coordinates_file}</coordinates_file>
    <lowpass_cutoff_frequency_for_coordinates>{lowpass_cutoff}</lowpass_cutoff_frequency_for_coordinates>
  </AnalyzeTool>
</OpenSimDocument>
"""


def run_static_optimization(
    participant_id: str,
    scaled_model_path: Path,
    ik_mot_path: Path,
    participant_output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    """
    Run OpenSim Static Optimization for one participant.

    Produces both activation and force .sto files by generating an
    AnalyzeTool XML setup file (the same approach the GUI uses) and
    executing it via the OpenSim Python API.

    Parameters
    ----------
    participant_id         : e.g. 'SHIVANGI'
    scaled_model_path      : Path to participant-specific scaled .osim model
    ik_mot_path            : Path to IK output .mot (used as kinematic input)
    participant_output_dir : Root output dir for this participant
    config                 : Global pipeline config dict

    Returns
    -------
    dict with keys:
      'activation_sto': Path to *_StaticOptimization_activation.sto
      'force_sto': Path to *_StaticOptimization_force.sto
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

    so_dir = get_analysis_dir(participant_output_dir, "so")
    so_cfg = config.get("static_optimization", {})
    activation_exponent = so_cfg.get("activation_exponent", 2)

    # ------------------------------------------------------------------
    # 1.  Load model.
    #
    #     FIX: Do NOT lock any coordinates by default (lock_coordinates: []).
    #     The GUI never locks pelvis/ankle/subtalar/mtp — it lets the model's
    #     built-in reserve actuators balance those unconstrained DOFs.
    #     Locking them caused muscle forces to be 70-82% lower than GUI
    #     because the muscles no longer needed to resist those joint moments.
    # ------------------------------------------------------------------
    model = osim.Model(str(scaled_model_path))

    coords_to_lock = so_cfg.get("lock_coordinates", [])
    coord_set = model.getCoordinateSet()
    locked_count = 0
    for cname in coords_to_lock:
        try:
            coord = coord_set.get(cname)
            coord.set_locked(True)
            locked_count += 1
        except Exception:
            logger.warning(
                "[%s] Coordinate '%s' not found in model — skipping lock",
                participant_id, cname,
            )

    if locked_count:
        logger.info(
            "[%s] Locked %d coordinates for SO: %s",
            participant_id, locked_count, coords_to_lock,
        )
    else:
        logger.info(
            "[%s] No coordinates locked — reserve actuators handle all DOFs (matches GUI).",
            participant_id,
        )

    model.initSystem()

    # Save model to SO output directory for reproducibility
    temp_model_path = so_dir / f"{participant_id}_so_model.osim"
    model.printToXML(str(temp_model_path))
    logger.debug("[%s] Saved model -> %s", participant_id, temp_model_path.name)

    # ------------------------------------------------------------------
    # 2.  Determine time range from .mot file.
    #
    #     FIX: clip_start_time=0.0 corrects the slight negative-time offset
    #     that some OpenCap .mot files carry (e.g. t_start = -3.18e-05 s).
    #     Without clipping, the AnalyzeTool places SO frames starting from
    #     that negative value, causing step_interval=4 to produce 254 output
    #     rows instead of the GUI's ~240 rows (dt~0.0667 s per row).
    # ------------------------------------------------------------------
    mot_df = read_mot_file(ik_mot_path)
    if "time" not in mot_df.columns:
        raise ValueError(f"No 'time' column found in {ik_mot_path}")

    clip_start = float(so_cfg.get("clip_start_time", 0.0))
    raw_t_start = float(mot_df["time"].iloc[0])
    t_start = max(raw_t_start, clip_start)
    t_end = float(mot_df["time"].iloc[-1])

    logger.info(
        "[%s] SO time range: %.4f -> %.4f s  (raw mot start was %.6f s)",
        participant_id, t_start, t_end, raw_t_start,
    )

    # ------------------------------------------------------------------
    # 3.  Write AnalyzeTool setup XML (mirrors GUI workflow)
    # ------------------------------------------------------------------
    use_force_set = str(so_cfg.get("use_model_force_set", True)).lower()
    lowpass_cutoff = so_cfg.get("lowpass_cutoff_frequency", -1)
    step_interval = so_cfg.get("step_interval", 4)

    setup_xml_content = _SETUP_XML_TEMPLATE.format(
        tool_name=participant_id,
        model_file=str(temp_model_path),
        results_dir=str(so_dir),
        t_start=t_start,
        t_end=t_end,
        activation_exponent=activation_exponent,
        use_model_force_set=use_force_set,
        coordinates_file=str(ik_mot_path),
        lowpass_cutoff=lowpass_cutoff,
        step_interval=step_interval,
    )
    logger.info(
        "[%s] SO settings: step_interval=%d, lowpass=%.1f Hz, "
        "use_model_force_set=%s, locked_coords=%s",
        participant_id, step_interval, lowpass_cutoff, use_force_set,
        coords_to_lock or "none (GUI-compatible)",
    )

    setup_xml_path = so_dir / f"{participant_id}_so_setup.xml"
    setup_xml_path.write_text(setup_xml_content, encoding="utf-8")
    logger.debug("[%s] Wrote SO setup XML -> %s", participant_id, setup_xml_path.name)

    # ------------------------------------------------------------------
    # 4.  Load and run AnalyzeTool from XML
    # ------------------------------------------------------------------
    logger.info(
        "[%s] Running Static Optimization (activation_exponent=%d)...",
        participant_id, activation_exponent,
    )
    logger.info(
        "[%s] SO may take 3-10 minutes depending on trial duration.", participant_id,
    )

    analyze_tool = osim.AnalyzeTool(str(setup_xml_path))
    analyze_tool.run()

    # ------------------------------------------------------------------
    # 5.  Locate output files
    # ------------------------------------------------------------------
    activation_sto = _find_so_output(so_dir, participant_id, "activation")
    force_sto = _find_so_output(so_dir, participant_id, "force")

    logger.info(
        "[%s] SO complete -> activation: %s, force: %s",
        participant_id, activation_sto.name, force_sto.name,
    )

    return {"activation_sto": activation_sto, "force_sto": force_sto}


def _find_so_output(so_dir: Path, participant_id: str, output_type: str) -> Path:
    """
    Find the SO output .sto file matching the given type ('activation' or 'force').
    OpenSim names them: <tool_name>_StaticOptimization_<type>.sto
    """
    pattern = f"*StaticOptimization_{output_type}.sto"
    matches = list(so_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Static Optimization {output_type} output not found in {so_dir}\n"
            f"Expected pattern: {pattern}\n"
            f"Files in directory: {[f.name for f in so_dir.iterdir()]}"
        )
    return matches[0]
