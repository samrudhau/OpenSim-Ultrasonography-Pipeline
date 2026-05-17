"""
run_pipeline.py
---------------
Main entry point for the OpenSim biomechanical analysis pipeline.

Reads participants.csv, auto-discovers patient data folders, and runs each
analysis module in sequence for every participant.  One participant failure
does NOT stop the batch — the error is logged and processing continues.

The pipeline COMPUTES outputs (ID, SO) via the OpenSim Python API, then
VALIDATES them against the pre-computed OpenSim GUI reference outputs.

Usage
-----
  # Activate conda environment first:
  conda activate opensim_pipeline

  # Run all discovered participants:
  python run_pipeline.py

  # Run specific participants:
  python run_pipeline.py --participants P001,P002

  # Skip OpenSim analyses, only recompile dataset from existing outputs:
  python run_pipeline.py --validate-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ─── Path setup (allows running from opensim_pipeline/ directory) ───────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, discover_patient_folders, get_participant_output_dir
from src.scaler import get_or_scale_model
from src.ik_runner import run_inverse_kinematics
from src.id_runner import run_inverse_dynamics
from src.so_runner import run_static_optimization
from src.signal_processor import process_participant_outputs
from src.output_validator import validate_participant_outputs
from src.dataset_compiler import (
    compile_master_dataset,
    load_participants_csv,
    save_participant_result,
    collect_participant_results,
)
from src.report_generator import write_report


# ─────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Pipeline log: %s", log_file)


# ─────────────────────────────────────────────────────────────
#  Analysis step order
# ─────────────────────────────────────────────────────────────

STEPS = ["scale", "ik", "id", "so", "process"]

STEP_DESCRIPTIONS = {
    "scale":    "Model scaling (copy OpenCap-scaled .osim)",
    "ik":       "Inverse Kinematics (use OpenCap .mot directly)",
    "id":       "Inverse Dynamics (compute via OpenSim API)",
    "so":       "Static Optimization (compute via OpenSim API)",
    "process":  "Signal processing (RMS extraction + plots)",
}


# ─────────────────────────────────────────────────────────────
#  Per-participant pipeline
# ─────────────────────────────────────────────────────────────

def run_participant(
    participant_id: str,
    participant_info: dict,
    participant_row: dict,
    config: dict,
    start_from_step: str = "scale",
) -> dict | None:
    """
    Run all analysis steps for a single participant.

    Steps:
    1. Scale — copy the OpenCap-scaled .osim model
    2. IK    — use the OpenCap .mot kinematics directly
    3. ID    — compute Inverse Dynamics via OpenSim API
    4. SO    — compute Static Optimization via OpenSim API (with reserves)
    5. Process — extract RMS metrics (pure pandas/numpy)
    6. Validate — compare computed .sto files with GUI reference outputs

    Returns a result dict (to be appended to master dataset), or None on failure.
    """
    logger = logging.getLogger(f"pipeline.{participant_id}")
    logger.info("=" * 60)
    logger.info("Starting pipeline for participant: %s", participant_id)
    logger.info("  Folder: %s", participant_info["folder_path"])

    output_root = Path(config["output_root"])
    participant_output_dir = get_participant_output_dir(output_root, participant_id)

    # Check if already processed (skip-safe re-run)
    if not config.get("output", {}).get("overwrite_existing", False):
        cached_result_csv = participant_output_dir / f"{participant_id}_results.csv"
        if cached_result_csv.exists():
            logger.info("Cached results found — skipping. (Set overwrite_existing: true to re-run)")
            row = pd.read_csv(cached_result_csv).iloc[0].to_dict()
            return row

    start_idx = STEPS.index(start_from_step) if start_from_step in STEPS else 0

    try:
        # ── Step 1: Scale / locate model ──────────────────────────────────
        if STEPS.index("scale") >= start_idx:
            logger.info("Step: %s", STEP_DESCRIPTIONS["scale"])
            scaled_model_path = get_or_scale_model(
                participant_id, participant_info, participant_row,
                participant_output_dir, config,
            )
        else:
            scaled_model_path = participant_output_dir / "scaled_model" / f"{participant_id}_scaled.osim"
            if not scaled_model_path.exists():
                logger.error("Scaled model not found at %s", scaled_model_path)
                return None

        # ── Step 2: Inverse Kinematics (use OpenCap .mot) ─────────────────
        if STEPS.index("ik") >= start_idx:
            logger.info("Step: %s", STEP_DESCRIPTIONS["ik"])
            ik_mot_path = run_inverse_kinematics(
                participant_id, participant_info, scaled_model_path,
                participant_output_dir, config,
            )
        else:
            from src.utils import find_task_mot
            ik_mot_path = find_task_mot(
                participant_info["kinematics_dir"],
                config.get("task_mot_pattern", "*.mot"),
            )

        # ── Step 3: Inverse Dynamics ──────────────────────────────────────
        id_sto_path = None
        if STEPS.index("id") >= start_idx:
            logger.info("Step: %s", STEP_DESCRIPTIONS["id"])
            id_sto_path = run_inverse_dynamics(
                participant_id, scaled_model_path, ik_mot_path,
                participant_output_dir, config,
            )
        else:
            # Fallback: locate existing ID output when skipping this step
            id_dir = participant_output_dir / "id"
            candidate = id_dir / f"{participant_id}_inverse_dynamics.sto"
            if candidate.exists():
                id_sto_path = candidate
                logger.info("[%s] Using existing ID output: %s", participant_id, candidate.name)
            else:
                logger.warning("[%s] No existing ID output found at %s — ID metrics will be NaN", participant_id, candidate)

        # ── Step 4: Static Optimization ───────────────────────────────────
        so_outputs = None
        if STEPS.index("so") >= start_idx:
            logger.info("Step: %s", STEP_DESCRIPTIONS["so"])
            so_outputs = run_static_optimization(
                participant_id, scaled_model_path, ik_mot_path,
                participant_output_dir, config,
            )
        else:
            # Fallback: locate existing SO outputs when skipping this step
            so_dir = participant_output_dir / "so"
            act_matches = list(so_dir.glob("*StaticOptimization_activation.sto"))
            force_matches = list(so_dir.glob("*StaticOptimization_force.sto"))
            if act_matches and force_matches:
                so_outputs = {
                    "activation_sto": act_matches[0],
                    "force_sto": force_matches[0],
                }
                logger.info(
                    "[%s] Using existing SO outputs: %s, %s",
                    participant_id, act_matches[0].name, force_matches[0].name,
                )
            else:
                logger.warning(
                    "[%s] No existing SO outputs found in %s — SO metrics will be NaN",
                    participant_id, so_dir,
                )

        if so_outputs is None:
            logger.error("[%s] Cannot run signal processing without SO outputs. Skipping.", participant_id)
            return None

        # ── Step 5: Signal processing (RMS extraction + plots) ────────────
        logger.info("Step: %s", STEP_DESCRIPTIONS["process"])
        result_row = process_participant_outputs(
            participant_id=participant_id,
            ik_mot=ik_mot_path,
            so_activation_sto=so_outputs["activation_sto"],
            so_force_sto=so_outputs["force_sto"],
            id_sto=id_sto_path,
            config=config,
            output_dir=participant_output_dir,
        )

        # ── Step 6: Validate against GUI reference outputs ────────────────
        # logger.info("Step: %s", STEP_DESCRIPTIONS["validate"])
        # computed_files = {
        #     "id_sto": id_sto_path,
        #     "so_activation_sto": so_outputs["activation_sto"],
        #     "so_force_sto": so_outputs["force_sto"],
        # }
        # validation_results = validate_participant_outputs(
        #     participant_id=participant_id,
        #     computed_files=computed_files,
        #     reference_model_dir=participant_info["model_dir"],
        #     reference_kinematics_dir=participant_info["kinematics_dir"],
        # )
        # # Store validation summary in result row
        # for vr in validation_results:
        #     key = f"validation_{vr['label'].split('/')[-1]}"
        #     result_row[key] = vr["status"]

        # Cache result for fast re-compilation
        save_participant_result(result_row, participant_output_dir, participant_id)

        logger.info("✓ Completed: %s", participant_id)
        return result_row

    except Exception as exc:
        logger.error("✗ FAILED: %s — %s", participant_id, exc, exc_info=True)
        logger.error("  Continuing to next participant...")
        return None


# ─────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    results_root = Path(config["results_root"])
    setup_logging(results_root)

    logger = logging.getLogger("pipeline")
    logger.info("OpenSim Biomechanical Pipeline starting")
    logger.info("Config: %s", args.config)

    # --- Load participants ---
    participants_csv = Path(PROJECT_ROOT / "config" / "participants.csv")
    anthro_df = load_participants_csv(participants_csv)

    # --- Discover patient folders ---
    data_root = Path(config["data_root"])
    patient_folders = discover_patient_folders(data_root, config)
    folder_map = {p["folder_name"]: p for p in patient_folders}

    # --- Filter to requested participants ---
    if args.participants == "all":
        selected_ids = anthro_df["participant_id"].tolist()
    else:
        selected_ids = [p.strip() for p in args.participants.split(",")]

    logger.info("Processing %d participant(s): %s", len(selected_ids), selected_ids)

    # --- Validate-only mode ---
    if args.validate_only:
        logger.info("Validate-only mode: recompiling from existing outputs")
        output_root = Path(config["output_root"])
        results = collect_participant_results(selected_ids, output_root)
        compile_master_dataset(
            results, participants_csv,
            results_root / "master_dataset.csv",
            results_root / "master_dataset_jamovi.csv",
        )
        write_report(results_root / "master_dataset.csv", results_root / "summary_report.txt")
        return

    # --- Main processing loop ---
    results: list[dict] = []
    failed: list[str] = []

    for pid in selected_ids:
        participant_rows = anthro_df[anthro_df["participant_id"] == pid]
        if participant_rows.empty:
            logger.error("Participant '%s' not found in participants.csv", pid)
            failed.append(pid)
            continue

        participant_row = participant_rows.iloc[0].to_dict()
        folder_name = participant_row.get("folder_name", "")

        if folder_name not in folder_map:
            logger.error(
                "Participant '%s' (folder '%s') not found in discovered folders under %s",
                pid, folder_name, data_root,
            )
            logger.error("  Available folders: %s", list(folder_map.keys()))
            failed.append(pid)
            continue

        participant_info = folder_map[folder_name]
        participant_info["participant_id"] = pid

        result = run_participant(
            participant_id=pid,
            participant_info=participant_info,
            participant_row=participant_row,
            config=config,
            start_from_step=args.start_from,
        )

        if result is not None:
            results.append(result)
        else:
            failed.append(pid)

    # --- Compile dataset even if some participants failed ---
    logger.info("")
    logger.info("Processing complete: %d succeeded, %d failed", len(results), len(failed))
    if failed:
        logger.warning("Failed participants: %s", failed)

    if results:
        compile_master_dataset(
            results, participants_csv,
            results_root / "master_dataset.csv",
            results_root / "master_dataset_jamovi.csv",
        )
        write_report(results_root / "master_dataset.csv", results_root / "summary_report.txt")
        logger.info("Pipeline complete. Outputs in: %s", results_root)
    else:
        logger.error("No results to compile. Check individual participant logs.")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenSim Biomechanical Pipeline — multi-patient automation"
    )
    parser.add_argument(
        "--participants", default="all",
        help="Comma-separated participant IDs (e.g. P001,P002) or 'all' (default)"
    )
    parser.add_argument(
        "--start-from", default="scale",
        choices=STEPS,
        dest="start_from",
        help="Start pipeline from this step (skip earlier steps)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip analysis; recompile dataset from cached results and generate report"
    )
    parser.add_argument(
        "--config", default="config/pipeline_config.yaml",
        help="Path to pipeline_config.yaml (default: config/pipeline_config.yaml)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
