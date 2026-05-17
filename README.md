# OpenSim Biomechanical Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20252586.svg)](https://doi.org/10.5281/zenodo.20252586)

Automated Python pipeline for multi-patient biomechanical analysis using OpenSim.
Replaces manual GUI steps for Inverse Dynamics, Static Optimization, and Joint
Reaction Analysis and compiles results into a single JAMOVI-ready CSV.

---

## Prerequisites

- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.10 (handled automatically by the environment)

> Note: This repository does not hardcode dataset locations. Set the root
> dataset folder in `config/pipeline_config.yaml` by editing the `data_root`
> value to point to the directory that contains your participant folders.

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

**Then open `config/participants.csv`** and fill in the demographic columns if
they are missing:
- `age`, `sex`, `dominant_hand`, `years_experience`
- Anthropometric measurements (height, weight, segment lengths)
- Work data (scanning hours, patients per day, etc.)

### Verify Muscle Names in Your Model

After activating the environment, list muscle names from the model with:

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

## Step 3 : Review the Config

Open `config/pipeline_config.yaml` and verify important settings. The most
important value to set before running the pipeline is `data_root` : it should
point to the folder that contains your participant subfolders.

Other useful settings:

- `use_opencap_kinematics` (default: `true`) : use OpenCap `.mot` kinematics
- `use_opencap_scaled_model` (default: `true`) : use OpenCap-produced scaled `.osim`
- `lowpass_filter_freq` (default: `6.0`) : low-pass filter frequency for kinematics
- `activation_exponent` (default: `2`) : exponent used in SO force model
- `save_plots` (default: `true`) : save per-participant activation plots
- `overwrite_existing` (default: `false`) : set to `true` to force re-runs

---

## Step 4 : Run the Pipeline

### Full run (all participants):

```bash
cd opensim_pipeline
conda activate opensim_pipeline
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

## Step 5 : Run Unit Tests

Verify the RMS calculations are correct before running the full pipeline:

```bash
cd opensim_pipeline
conda activate opensim_pipeline
pytest tests/ -v
```

The integration tests will automatically use existing example outputs when available
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
| SO/JRA output is all zeros | Check `opensim.log` in the participant's output folder |
| Muscle name not found warning | Run `python setup_participants.py --list-muscles` and update config |
| IK residual warning (>2 cm) | Check marker placement in the static trial |
| JRA produces NaN | Verify SO force `.sto` was produced successfully before JRA |

---

## Project Structure

```
opensim_pipeline/
├── config/
│   ├── participants.csv          # One row per participant (generated)
│   └── pipeline_config.yaml      # Global analysis settings (edit `data_root`)
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
