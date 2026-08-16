# OpenSim Ultrasonography Pipeline — BUET Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20252586.svg)](https://doi.org/10.5281/zenodo.20252586)

Automated Python pipeline for multi-patient biomechanical analysis of sonographers
using OpenSim and the **Bilateral Upper Extremity Trunk (BUET) model**.

Replaces manual GUI steps for Inverse Kinematics, Inverse Dynamics, Static
Optimization, and Joint Reaction Analysis — and compiles results into a single
JAMOVI-ready CSV for statistical analysis.

> **v2.0.0 — BUET Model Integration**
> This release switches from the Lai-Uhlrich lower-body model to the BUET model,
> enabling bilateral shoulder, elbow, and wrist analysis without re-scaling.
> See [CHANGELOG](#changelog) below.

---

## Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10 (handled automatically by the environment)

> **Note:** This repository does not include participant data or model files.
> Set the root dataset folder in `config/pipeline_config.yaml` by editing the
> `data_root` value to point to the directory that contains your participant folders.
> Each participant folder must contain `OpenSimData/Model/Bilateral Upper Extremity
> Trunk Model.osim` and `OpenSimData/Kinematics/usg.mot`.

---

## Step 1 : Create the Conda Environment (one-time setup)

Open a terminal and run the following from the repository root:

```bash
cd opensim_pipeline
conda env create -f environment.yml
conda activate opensim_pipeline
```

This installs OpenSim + all dependencies. It may take several minutes.

---

## Step 2 : Set Up Participants

Run the setup script to auto-discover all patient folders and build `participants.csv`.
Make sure you have set `data_root` in `config/pipeline_config.yaml` before running.

```bash
conda activate opensim_pipeline
cd opensim_pipeline
python setup_participants.py
```

The script scans the directory configured in `data_root` for folders containing
`MarkerData/` and `OpenSimData/` and adds them to `config/participants.csv`.

**Then open `config/participants.csv`** and fill in the demographic columns:
- `age`, `sex`, `dominant_hand`, `years_experience`
- Anthropometric measurements (height, weight, segment lengths)
- Work data (scanning hours, patients per day, etc.)

### Verify Muscle and Joint Names (BUET Model)

```bash
python setup_participants.py --list-muscles   # Print all BUET model muscle names
python setup_participants.py --list-joints    # Print all BUET model coordinate names
```

Compare against the `muscle_groups`, `joint_dof_map`, and `joint_angle_map`
sections in `config/pipeline_config.yaml` and update if needed.

---

## Step 3 : Review the Config

Open `config/pipeline_config.yaml`. Key settings for the BUET model workflow:

| Setting | Default | Description |
|---|---|---|
| `data_root` | *(set this)* | Root folder containing participant subfolders |
| `model.use_buet_model` | `true` | Use BUET model directly (no scaling) |
| `model.buet_model_name` | `Bilateral Upper Extremity Trunk Model.osim` | Filename in each participant's `Model/` folder |
| `inverse_kinematics.use_opencap_kinematics` | `true` | Use pre-computed OpenCap `.mot` |
| `lumbar_coord_rename` | *(configured)* | Renames 3 lumbar coords to match BUET model |
| `joint_reaction_analysis.joints_of_interest` | 8 joints | Glenohumeral, elbow, radioulnar, wrist (bilateral) |
| `lowpass_filter_freq` | `6.0` | Low-pass filter for kinematics/dynamics |
| `overwrite_existing` | `true` | Force re-run of existing outputs |

---

## Step 4 : Run the Pipeline

### Full run (all participants):

```bash
conda activate opensim_pipeline
cd opensim_pipeline
python run_pipeline.py
```

### Specific participants only:

```bash
python run_pipeline.py --participants AKSHITHA
python run_pipeline.py --participants AKSHITHA,SHIVANGI
```

### Start from a specific step (skip earlier steps):

```bash
python run_pipeline.py --start-from scale    # Full run (default)
python run_pipeline.py --start-from ik       # Re-adapt .mot + ID + SO + JRA + process
python run_pipeline.py --start-from id       # Re-run ID, SO, JRA, and processing
python run_pipeline.py --start-from so       # Re-run SO, JRA, and processing
python run_pipeline.py --start-from jra      # Re-run JRA and processing only
python run_pipeline.py --start-from process  # Only recompute RMS + plots
```

### Recompile dataset without re-running analysis:

```bash
python run_pipeline.py --validate-only
```

---

## Step 5 : Run Unit Tests

```bash
conda activate opensim_pipeline
pytest tests/ -v
```

---

## Output Files

After a successful run:

| File | Description |
|------|-------------|
| `results/master_dataset.csv` | One row per participant, all biomechanical variables |
| `results/master_dataset_jamovi.csv` | Same data, JAMOVI-compatible column names |
| `results/summary_report.txt` | Descriptive statistics (median, IQR, min, max) per variable |
| `results/pipeline_run_*.log` | Full execution log with timestamps |
| `outputs/<PATIENT>/mot_adapted/` | BUET-adapted `.mot` (lumbar coords renamed) |
| `outputs/<PATIENT>/id/` | Inverse Dynamics `.sto` files |
| `outputs/<PATIENT>/so/` | Static Optimization `.sto` files (activation + force) |
| `outputs/<PATIENT>/jra/` | Joint Reaction Analysis `.sto` files |
| `outputs/<PATIENT>/plots/` | Per-participant muscle activation PNG plots |

### Import to JAMOVI

Open JAMOVI → File → Open → `results/master_dataset_jamovi.csv`

All columns use JAMOVI-safe names (no slashes, spaces, or special characters).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named opensim` | Run `conda activate opensim_pipeline` first |
| `BUET model not found in ...` | Check that each participant's `OpenSimData/Model/` contains `Bilateral Upper Extremity Trunk Model.osim` |
| Participant not found in pipeline | Run `python setup_participants.py` to add it |
| SO/JRA output is all zeros or fails | Check `opensim.log` in the participant's output folder |
| Muscle name not found warning | Run `--list-muscles` and update `muscle_groups` in config |
| JRA produces NaN | Verify SO force `.sto` was produced successfully before JRA |
| Lumbar coordinates not matching | Check `lumbar_coord_rename` in config matches your model's coordinate names |

---

## Project Structure

```
opensim_pipeline/
├── config/
│   ├── participants.csv          # One row per participant (generated by setup_participants.py)
│   └── pipeline_config.yaml      # Global analysis settings (edit data_root + model settings)
├── src/
│   ├── utils.py                 # File parsers, folder discovery
│   ├── mot_adapter.py           # Renames lumbar coords in .mot for BUET model compatibility
│   ├── scaler.py                # Model acquisition (copies BUET model, no scaling)
│   ├── ik_runner.py             # Inverse Kinematics (or use OpenCap .mot directly)
│   ├── id_runner.py             # Inverse Dynamics
│   ├── so_runner.py             # Static Optimization
│   ├── jra_runner.py            # Joint Reaction Analysis (bilateral shoulder/elbow/wrist)
│   ├── signal_processor.py      # RMS extraction, JRA resultant force computation
│   ├── dataset_compiler.py      # Master CSV assembly
│   └── report_generator.py      # Summary report + cohort spread validation
├── tests/
│   └── test_signal_processor.py
├── outputs/                     # Per-patient analysis outputs (auto-created, not tracked)
├── results/                     # Master dataset + report (auto-created, not tracked)
├── CITATION.cff                 # Citation metadata (GitHub "Cite this repository")
├── .zenodo.json                 # Zenodo upload metadata
├── environment.yml              # Conda environment definition
├── run_pipeline.py              # Main entry point
└── setup_participants.py        # Setup helper
```

---

## Changelog

### v2.0.0 (2026-08-16) — BUET Model Integration

- **New**: `src/mot_adapter.py` — renames 3 lumbar DOF columns in OpenCap `.mot` files to match BUET model coordinate names (`lumbar_extension` → `flex_extension`, etc.)
- **New**: BUET model mode in `src/scaler.py` — copies `Bilateral Upper Extremity Trunk Model.osim` from participant's folder without re-scaling
- **Updated**: `src/jra_runner.py` — rewritten with XML-based `AnalyzeTool` (same approach as SO runner); now correctly produces output for bilateral glenohumeral, elbow, radioulnar, and wrist joints
- **Updated**: `run_pipeline.py` — 6-step pipeline: `scale → ik → id → so → jra → process`; JRA is now a first-class step
- **Updated**: `src/signal_processor.py` — new `extract_jra_rms()` computes 3D resultant force RMS per joint
- **Updated**: `config/pipeline_config.yaml` — full bilateral UE muscle group config (deltoid, rotator cuff, elbow flexors/extensors, pronators, lumbar erectors, latissimus dorsi) and JRA joint map
- **Updated**: `src/dataset_compiler.py` — bilateral column order for both sides
- **Updated**: `src/report_generator.py` — dynamic cohort-median validation (replaces hardcoded Lai-Uhlrich thesis medians)

### v1.x — Lai-Uhlrich Model (archived)

- Original pipeline using OpenCap-scaled Lai-Uhlrich model
- ID, SO for lower-body and single-arm analysis

---

## Citation

If you use this pipeline in your research, please cite:

```
Samrudhau. (2026). OpenSim Ultrasonography Pipeline — BUET Model (v2.0.0).
Zenodo. https://doi.org/10.5281/zenodo.20252586
```

Or use the `CITATION.cff` file (GitHub "Cite this repository" button).

---

## License

MIT License. See [LICENSE](LICENSE).
