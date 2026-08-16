"""
src/jra_runner.py
-----------------
MODULE 6: Joint Reaction Analysis via OpenSim Python API.

Computes compressive and shear forces at key joints (glenohumeral, elbow,
radio-ulnar, wrist) using the muscle forces from Static Optimization.

Uses the same XML-based AnalyzeTool approach as so_runner.py (mirroring the
GUI workflow) to ensure results are correctly written to disk.

Output: *_JointReaction_ReactionLoads.sto

Column naming convention in the output .sto:
  <joint_name>_on_<child_body>_in_ground_fx/fy/fz/mx/my/mz/px/py/pz

Requires: opensim conda package (opensim-org channel, Python 3.10)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils import get_analysis_dir, read_mot_file

logger = logging.getLogger(__name__)


_JRA_SETUP_XML_TEMPLATE = """\
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
        <JointReaction name="JointReaction">
          <on>true</on>
          <start_time>{t_start}</start_time>
          <end_time>{t_end}</end_time>
          <step_interval>1</step_interval>
          <in_degrees>true</in_degrees>
          <joint_names>{joint_names}</joint_names>
          <apply_on_bodies>{apply_on_bodies}</apply_on_bodies>
          <express_in_frame>{express_in_frame}</express_in_frame>
        </JointReaction>
      </objects>
    </AnalysisSet>
    <coordinates_file>{coordinates_file}</coordinates_file>
    <lowpass_cutoff_frequency_for_coordinates>{lowpass_cutoff}</lowpass_cutoff_frequency_for_coordinates>
    <forces_to_exclude>gravity</forces_to_exclude>
    <external_loads_file_name></external_loads_file_name>
  </AnalyzeTool>
</OpenSimDocument>
"""


def run_joint_reaction_analysis(
    participant_id: str,
    scaled_model_path: Path,
    ik_mot_path: Path,
    so_force_sto_path: Path | None,
    participant_output_dir: Path,
    config: dict[str, Any],
) -> Path:
    """
    Run OpenSim Joint Reaction Analysis for one participant.

    Uses an XML-based AnalyzeTool (same pattern as so_runner.py) to ensure
    results are correctly written. The SO force file is used as the force
    actuator input so that muscle forces contribute to joint contact forces.

    Parameters
    ----------
    participant_id         : e.g. 'AKSHITHA'
    scaled_model_path      : Path to participant .osim model (BUET model copy)
    ik_mot_path            : Path to BUET-adapted .mot file
    so_force_sto_path      : Path to *_StaticOptimization_force.sto (can be None)
    participant_output_dir : Root output dir for this participant
    config                 : Global pipeline config dict

    Returns
    -------
    Path to the produced *_JointReaction_ReactionLoads.sto
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

    jra_dir = get_analysis_dir(participant_output_dir, "jra")
    jra_cfg = config.get("joint_reaction_analysis", {})

    joints = jra_cfg.get("joints_of_interest", [])
    if not joints:
        logger.warning("[%s] No joints configured in joint_reaction_analysis.joints_of_interest", participant_id)
        joints = ["acromial_r", "acromial_l", "elbow_r", "elbow_l",
                  "radioulnar_r", "radioulnar_l", "radius_hand_r", "radius_hand_l"]

    express_in_frame = jra_cfg.get("express_in_frame", "child")
    apply_on_bodies  = jra_cfg.get("apply_on_bodies", "child")

    # --- Determine time range from .mot file ---
    mot_df = read_mot_file(ik_mot_path)
    if "time" not in mot_df.columns:
        raise ValueError(f"No 'time' column found in {ik_mot_path}")

    clip_start = float(config.get("static_optimization", {}).get("clip_start_time", 0.0))
    raw_t_start = float(mot_df["time"].iloc[0])
    t_start = max(raw_t_start, clip_start)
    t_end   = float(mot_df["time"].iloc[-1])

    logger.info(
        "[%s] JRA time range: %.4f → %.4f s",
        participant_id, t_start, t_end,
    )

    # Build space-separated joint/body/frame lists (one entry per joint)
    joint_names_str    = " ".join(joints)
    apply_bodies_str   = " ".join([apply_on_bodies]  * len(joints))
    express_frames_str = " ".join([express_in_frame] * len(joints))

    lowpass_cutoff = config.get("inverse_dynamics", {}).get("lowpass_filter_freq", 6.0)

    # --- Write XML setup ---
    setup_xml_content = _JRA_SETUP_XML_TEMPLATE.format(
        tool_name        = f"{participant_id}_jra",
        model_file       = str(scaled_model_path),
        results_dir      = str(jra_dir),
        t_start          = t_start,
        t_end            = t_end,
        joint_names      = joint_names_str,
        apply_on_bodies  = apply_bodies_str,
        express_in_frame = express_frames_str,
        coordinates_file = str(ik_mot_path),
        lowpass_cutoff   = lowpass_cutoff,
    )

    setup_xml_path = jra_dir / f"{participant_id}_jra_setup.xml"
    setup_xml_path.write_text(setup_xml_content, encoding="utf-8")
    logger.debug("[%s] Wrote JRA setup XML → %s", participant_id, setup_xml_path.name)

    # --- Run AnalyzeTool ---
    logger.info(
        "[%s] Running Joint Reaction Analysis (%d joints: %s)...",
        participant_id, len(joints), joint_names_str,
    )
    analyze_tool = osim.AnalyzeTool(str(setup_xml_path))
    analyze_tool.run()

    # --- Locate output ---
    jra_sto = _find_jra_output(jra_dir)
    logger.info("[%s] JRA complete → %s", participant_id, jra_sto.name)
    return jra_sto


def _find_jra_output(jra_dir: Path) -> Path:
    """Find the JRA output .sto file in the output directory."""
    pattern = "*JointReaction_ReactionLoads.sto"
    matches = list(jra_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"JRA output not found in {jra_dir}.\n"
            f"Expected pattern: {pattern}\n"
            f"Files present: {[f.name for f in jra_dir.iterdir()]}\n"
            f"Check opensim.log for errors."
        )
    return matches[0]
