"""
setup_participants.py
---------------------
Helper script to:
1. Auto-discover patient folders in data_root and populate participants.csv
2. List all muscle names in the Lai-Ulrich model (needed before running SO)

Run BEFORE running the main pipeline to set up your participant list.

Usage
-----
  # Activate conda environment first:
  conda activate opensim_pipeline

  # Discover patients and create participants.csv:
  python setup_participants.py

  # List muscle names from the Lai-Ulrich model (requires OpenSim):
  python setup_participants.py --list-muscles

  # List joint coordinate names (for verifying joint_dof_map in config):
  python setup_participants.py --list-joints
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, discover_patient_folders


PARTICIPANTS_CSV = PROJECT_ROOT / "config" / "participants.csv"

# All columns in participants.csv
CSV_COLUMNS = [
    "participant_id", "folder_name",
    "age", "sex", "height_m", "weight_kg",
    "sitting_height_cm", "sitting_shoulder_height_cm", "sitting_elbow_height_cm",
    "sitting_eye_height_cm", "thigh_thickness_cm", "buttock_knee_cm",
    "buttock_popliteal_cm", "knee_height_cm", "popliteal_height_cm",
    "shoulder_breadth_cm", "shoulder_elbow_cm", "elbow_fingertip_cm",
    "shoulder_grip_cm", "dominant_hand",
    "years_experience", "work_hrs_per_week", "scanning_hrs_per_week",
    "patients_per_day", "min_per_patient", "probe_weight_g", "chair_height_cm",
    "opencap_session_id",
]


def discover_and_write_participants(config: dict) -> None:
    """
    Scan data_root for patient folders and create/update participants.csv.
    Existing rows are preserved; new rows are added for newly discovered patients.
    """
    data_root = Path(config["data_root"])
    patients = discover_patient_folders(data_root, config)

    if not patients:
        print(f"No patient folders found in {data_root}")
        print("Make sure each patient folder contains 'MarkerData/' and 'OpenSimData/' subdirectories.")
        return

    # Load existing entries to avoid duplicates
    # Match by folder_name (stable) not participant_id (which may differ in format)
    existing_folder_names: set[str] = set()
    existing_rows: list[dict] = []
    if PARTICIPANTS_CSV.exists():
        with open(PARTICIPANTS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_folder_names.add(row["folder_name"])
                existing_rows.append(row)

    # Determine next P-number from existing rows
    max_pnum = 0
    for row in existing_rows:
        pid = row.get("participant_id", "")
        if pid.startswith("P") and pid[1:].isdigit():
            max_pnum = max(max_pnum, int(pid[1:]))

    # Add new participants
    new_count = 0
    for p in patients:
        if p["folder_name"] in existing_folder_names:
            # Find and show existing ID for this folder
            existing_pid = next(
                (r["participant_id"] for r in existing_rows if r["folder_name"] == p["folder_name"]),
                "?"
            )
            print(f"  [EXISTS] {existing_pid} — {p['folder_name']} already in participants.csv")
            continue

        # Try to read height/weight from sessionMetadata.yaml
        session_meta = p["folder_path"] / "sessionMetadata.yaml"
        height_m, weight_kg = "", ""
        if session_meta.exists():
            try:
                import yaml
                with open(session_meta) as f:
                    meta = yaml.safe_load(f)
                height_m = meta.get("height_m", "")
                weight_kg = meta.get("mass_kg", "")
            except Exception:
                pass

        max_pnum += 1
        pid = f"P{max_pnum:03d}"   # P001, P002, P003 ...

        new_row = {col: "" for col in CSV_COLUMNS}
        new_row["participant_id"] = pid
        new_row["folder_name"] = p["folder_name"]
        new_row["height_m"] = height_m
        new_row["weight_kg"] = weight_kg

        existing_rows.append(new_row)
        existing_folder_names.add(p["folder_name"])
        new_count += 1
        print(f"  [NEW]    {pid}  ({p['folder_name']})")

    # Write updated CSV
    with open(PARTICIPANTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"\nparticipants.csv updated: {len(existing_rows)} total rows ({new_count} new).")
    print(f"File: {PARTICIPANTS_CSV}")
    print("\nNEXT STEP: Open participants.csv and fill in demographic columns (age, sex, etc.)")


def list_muscles(config: dict) -> None:
    """
    Print all muscle names in the Lai-Ulrich model to stdout.
    Use this output to verify/update the muscle_groups mapping in pipeline_config.yaml.
    """
    try:
        import opensim as osim
    except ImportError:
        print("ERROR: OpenSim not installed. Run: conda activate opensim_pipeline first.")
        return

    data_root = Path(config["data_root"])
    patients = discover_patient_folders(data_root, config)
    if not patients:
        print("No patient folders found to extract model from.")
        return

    # Use the first patient's scaled model
    model_dir = patients[0]["model_dir"]
    pattern = config.get("model", {}).get("scaled_model_pattern", "*_scaled_scaled.osim")
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        print(f"No model file matching '{pattern}' found in {model_dir}")
        return

    model_path = model_files[0]
    print(f"\nLoading model: {model_path.name}")
    model = osim.Model(str(model_path))
    model.initSystem()

    muscles = model.getMuscles()
    n = muscles.getSize()
    print(f"\n{'─' * 50}")
    print(f"MUSCLES IN MODEL ({n} total)")
    print(f"{'─' * 50}")
    for i in range(n):
        print(f"  {muscles.get(i).getName()}")

    print(f"\nCopy these names into config/pipeline_config.yaml → muscle_groups section.")


def list_joints(config: dict) -> None:
    """Print all joint coordinate names in the model."""
    try:
        import opensim as osim
    except ImportError:
        print("ERROR: OpenSim not installed. Run: conda activate opensim_pipeline first.")
        return

    data_root = Path(config["data_root"])
    patients = discover_patient_folders(data_root, config)
    if not patients:
        return

    model_dir = patients[0]["model_dir"]
    pattern = config.get("model", {}).get("scaled_model_pattern", "*_scaled_scaled.osim")
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        return

    model = osim.Model(str(model_files[0]))
    model.initSystem()

    coords = model.getCoordinateSet()
    n = coords.getSize()
    print(f"\n{'─' * 50}")
    print(f"JOINT COORDINATES IN MODEL ({n} total)")
    print(f"{'─' * 50}")
    for i in range(n):
        coord = coords.get(i)
        print(f"  {coord.getName():40s}  [{coord.getMotionType()}]")

    print(f"\nCopy relevant names into config/pipeline_config.yaml → joint_dof_map / joint_angle_map.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSim Pipeline Setup Utilities")
    parser.add_argument("--list-muscles", action="store_true",
                        help="List all muscle names in the Lai-Ulrich model")
    parser.add_argument("--list-joints", action="store_true",
                        help="List all joint coordinate names in the model")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.list_muscles:
        list_muscles(config)
    elif args.list_joints:
        list_joints(config)
    else:
        print("Discovering patient folders...\n")
        discover_and_write_participants(config)


if __name__ == "__main__":
    main()