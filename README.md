# OpenSim Biomechanical Pipeline

Automated Python pipeline for multi-patient biomechanical analysis using OpenSim.
Replaces manual GUI steps for Inverse Dynamics, Static Optimization, and Joint
Reaction Analysis — and compiles results into a single JAMOVI-ready CSV.

---

## Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10 (handled automatically by the environment)
- Patient data folders already downloaded from OpenCap under `D:\samrudh\`

---

## Step 1 — Create the Conda Environment (one-time setup)

Open **Anaconda Prompt** (not regular PowerShell) and run:

```bash
cd "D:\OPENCAP FILES - Copy\opensim_pipeline"
conda env create -f environment.yml
```

This installs OpenSim + all dependencies. It may take 5–10 minutes.

To activate the environment every time before using the pipeline:

```bash
conda activate opensim_pipeline
```

---

## Step 2 — Set Up Participants

Run the setup script to auto-discover all patient folders and build `participants.csv`:

```bash
conda activate opensim_pipeline
cd "D:\OPENCAP FILES - Copy\opensim_pipeline"
python setup_participants.py
```

This scans `D:\OPENCAP FILES\` for folders containing 
`MarkerData\` and `OpenSimData\`
and adds them to `config/participants.csv`.

**Then open `config/participants.csv`** and fill in the demographic columns:
- `age`, `sex`, `dominant_hand`, `years_experience`
- Anthropometric measurements (height, weight, segment lengths)
- Work data (scanning hours, patients per day, etc.)

### Verify Muscle Names in Your Model

After activating the environment, run:

```bash
python setup_participants.py --list-muscles
```

Compare the printed muscle names against the `muscle_groups` section in
`config/pipeline_config.yaml` and update as needed.

```bash
python setup_participants.py --list-joints
```

Compare joint coordinate names against `joint_dof_map` and `joint_angle_map` in the config.

---

## Step 3 — Review the Config

Open `config/pipeline_config.yaml` and verify:

| Setting | Default | Notes |
|---------|---------|-------|
| `data_root` | `D:/samrudh` | Root of all patient folders |
| `use_opencap_kinematics` | `true` | Uses OpenCap .mot directly (recommended) |
| `use_opencap_scaled_model` | `true` | Uses OpenCap .osim (recommended) |
| `lowpass_filter_freq` | `6.0` Hz | Applied to kinematics before ID |
| `activation_exponent` | `2` | Thesis setting for SO |
| `save_plots` | `true` | Per-participant muscle activation plots |
| `overwrite_existing` | `false` | Safe re-run; skip already-processed patients |

---

## Step 4 — Run the Pipeline

### Full run (all participants):
```bash
conda activate opensim_pipeline
cd "D:\OPENCAP FILES - Copy\opensim_pipeline"
python run_pipeline.py
```

### Specific participants only:
```bash
python run_pipeline.py --participants P001
```

### Start from a specific step (skip earlier steps):
```bash
python run_pipeline.py --start-from so       # re-run SO, JRA, processing
python run_pipeline.py --start-from jra      # re-run JRA and processing only
python run_pipeline.py --start-from process  # only recompute RMS + plots
```

### Recompile dataset without re-running analysis:
```bash
python run_pipeline.py --validate-only
```

---

## Step 5 — Run Unit Tests

Verify the RMS calculations are correct before running the full pipeline:

```bash
conda activate opensim_pipeline
cd "D:\OPENCAP FILES - Copy\opensim_pipeline"
pytest tests/ -v
```

The integration tests will automatically use Shivangi's existing output files
to validate the `.sto` file parser and RMS extraction.

---

## Output Files

After a successful run:

| File | Description |
|------|-------------|
| `results/master_dataset.csv` | One row per participant, all biomechanical variables |
| `results/master_dataset_jamovi.csv` | Same data, JAMOVI-compatible column names |
| `results/summary_report.txt` | Descriptive statistics (median, IQR, min, max) |
| `results/pipeline_run_*.log` | Full execution log with timestamps |
| `outputs/<PATIENT>/plots/` | Per-participant muscle activation PNG plots |
| `outputs/<PATIENT>/id/` | Inverse Dynamics .sto files |
| `outputs/<PATIENT>/so/` | Static Optimization .sto files |
| `outputs/<PATIENT>/jra/` | Joint Reaction Analysis .sto files |

### Import to JAMOVI

Open JAMOVI → File → Open → `results/master_dataset_jamovi.csv`

All columns use JAMOVI-safe names (no slashes, spaces, or special characters).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named opensim` | Run `conda activate opensim_pipeline` first |
| Patient not found in pipeline | Run `python setup_participants.py` to add it |
| SO/JRA output is all zeros | Check opensim.log in the participant's output folder |
| Muscle name not found warning | Run `python setup_participants.py --list-muscles` and update config |
| IK residual warning (>2 cm) | Check marker placement in the static trial |
| JRA produces NaN | Verify SO force .sto was produced successfully before JRA |

---

## Project Structure

```
opensim_pipeline/
├── config/
│   ├── participants.csv          # One row per participant
│   └── pipeline_config.yaml     # Global analysis settings
├── src/
│   ├── utils.py                 # File parsers, folder discovery
│   ├── signal_processor.py      # RMS + rectification (no OpenSim needed)
│   ├── scaler.py                # Model scaling (uses OpenCap model by default)
│   ├── ik_runner.py             # Inverse Kinematics
│   ├── id_runner.py             # Inverse Dynamics
│   ├── so_runner.py             # Static Optimization
│   ├── jra_runner.py            # Joint Reaction Analysis
│   ├── dataset_compiler.py      # Master CSV assembly
│   └── report_generator.py      # Summary report + thesis validation
├── tests/
│   └── test_signal_processor.py # Unit + integration tests
├── outputs/                     # Per-patient analysis outputs (auto-created)
├── results/                     # Master dataset + report (auto-created)
├── environment.yml              # Conda environment definition
├── run_pipeline.py              # Main entry point
└── setup_participants.py        # Setup helper
```
