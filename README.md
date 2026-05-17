# OpenSim Biomechanical Pipeline

Automated Python pipeline for multi-participant biomechanical analysis using
OpenSim. The project provides utilities to discover participant data, prepare
scaled models, run OpenSim analyses (Inverse Dynamics, Static Optimization,
Joint Reaction Analysis), and compile results into a single analysis-ready CSV.

This README focuses on using the code in this repository. All dataset paths
are configured via `config/pipeline_config.yaml` — there are no hardcoded
absolute paths in the codebase.

---

## Quickstart

1. Create and activate the conda environment (one-time):

```bash
cd opensim_pipeline
conda env create -f environment.yml
conda activate opensim_pipeline
```

2. Configure your data root in `config/pipeline_config.yaml` by setting the
	 `data_root` value to the folder that contains your participant subfolders.

3. Discover participants and create `config/participants.csv`:

```bash
python setup_participants.py
```

4. Run the pipeline for all participants:

```bash
python run_pipeline.py
```

See `python run_pipeline.py --help` for options to run a subset of participants
or to start from a specific analysis step.

---

## Configuration

- `config/pipeline_config.yaml`: main project configuration. Set `data_root` to
	the directory containing participant folders. The pipeline expects each
	participant folder to contain `MarkerData/` and `OpenSimData/` (the
	`setup_participants.py` helper will validate this).
- `config/participants.csv`: populated by `setup_participants.py`; fill in any
	missing demographic or anthropometric fields before running OpenSim analyses.

---

## Running tests

Unit tests are under `tests/`. Run them with:

```bash
pytest tests/ -v
```

---

## Outputs

By default the pipeline writes computed files to `outputs/` (per-participant)
and summary datasets to `results/`. These folders are ignored by version
control and will be created at runtime.

---

## Project Structure

```
opensim_pipeline/
├── config/                # pipeline settings + participants.csv
├── src/                   # analysis modules and utilities
├── tests/                 # unit + integration tests
├── outputs/               # generated per-participant outputs (ignored)
├── results/               # compiled datasets and reports (ignored)
├── environment.yml        # conda environment
├── run_pipeline.py        # main entry point
└── setup_participants.py  # discover participants and populate CSV
```

---

If you'd like, I can further expand this README with examples, expected
folder layout for a single participant, or a short example dataset. Tell me
which you'd prefer.
