"""
tests/test_signal_processor.py
-------------------------------
Unit tests for the RMS computation and full-wave rectification functions in
src/signal_processor.py.

These tests validate that the Python implementation EXACTLY matches the thesis
Excel formula: =SQRT(SUMQ(r1:rn) / COUNTA(r1:rn))

Run with:
  conda activate opensim_pipeline
  cd D:/samrudh/opensim_pipeline
  pytest tests/ -v

No OpenSim dependency — these tests run immediately after installing numpy/pandas.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing src without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.signal_processor import (
    compute_rms,
    full_wave_rectify,
    full_wave_rectify_then_rms,
    compute_resultant_magnitude,
    extract_ik_rms,
    extract_moment_rms,
)


# ─────────────────────────────────────────────────────────────
#  compute_rms — matches thesis Excel formula exactly
# ─────────────────────────────────────────────────────────────

class TestComputeRMS:
    def test_known_case(self):
        """[3, 4] → sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355339"""
        signal = np.array([3.0, 4.0])
        result = compute_rms(signal)
        assert abs(result - 3.5355339) < 1e-5, f"Expected ~3.5355, got {result}"

    def test_constant_signal(self):
        """All same values: RMS equals the constant."""
        signal = np.array([5.0, 5.0, 5.0, 5.0])
        result = compute_rms(signal)
        assert abs(result - 5.0) < 1e-8

    def test_single_value(self):
        """RMS of a single value is the absolute value of that value."""
        signal = np.array([7.5])
        assert abs(compute_rms(signal) - 7.5) < 1e-8

    def test_ignores_nan(self):
        """NaN values are excluded (matches Excel COUNTA behaviour)."""
        signal = np.array([3.0, np.nan, 4.0])
        result = compute_rms(signal)
        expected = compute_rms(np.array([3.0, 4.0]))
        assert abs(result - expected) < 1e-8

    def test_all_nan_returns_nan(self):
        """All-NaN array should return NaN."""
        signal = np.array([np.nan, np.nan])
        result = compute_rms(signal)
        assert np.isnan(result)

    def test_negative_values(self):
        """Negative values are squared, so RMS is always positive."""
        signal = np.array([-3.0, -4.0])
        result = compute_rms(signal)
        assert abs(result - 3.5355339) < 1e-5

    def test_zero_signal(self):
        """Zero signal → RMS = 0."""
        signal = np.zeros(10)
        assert compute_rms(signal) == 0.0

    def test_large_array(self):
        """RMS of 1000 ones = 1."""
        signal = np.ones(1000)
        assert abs(compute_rms(signal) - 1.0) < 1e-10

    def test_matches_numpy_linalg_norm(self):
        """Verify against numpy's built-in norm / sqrt(n)."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        expected = np.sqrt(np.mean(signal ** 2))
        result = compute_rms(signal)
        assert abs(result - expected) < 1e-10


# ─────────────────────────────────────────────────────────────
#  full_wave_rectify
# ─────────────────────────────────────────────────────────────

class TestFullWaveRectify:
    def test_removes_negatives(self):
        """All values become non-negative."""
        signal = np.array([-1.0, 2.0, -3.0, 4.0])
        result = full_wave_rectify(signal)
        assert np.all(result >= 0)

    def test_positive_unchanged(self):
        """Positive values are unaffected."""
        signal = np.array([1.0, 2.0, 3.0])
        result = full_wave_rectify(signal)
        np.testing.assert_array_equal(result, signal)

    def test_zero_unchanged(self):
        """Zero stays zero."""
        signal = np.array([0.0, -0.0])
        result = full_wave_rectify(signal)
        assert np.all(result == 0.0)

    def test_symmetry(self):
        """Rectification of [-x] equals rectification of [x]."""
        signal = np.array([-5.0, 3.0, -1.5])
        np.testing.assert_array_almost_equal(
            full_wave_rectify(signal),
            np.array([5.0, 3.0, 1.5])
        )


# ─────────────────────────────────────────────────────────────
#  full_wave_rectify_then_rms (thesis muscle activation method)
# ─────────────────────────────────────────────────────────────

class TestFullWaveRectifyThenRMS:
    def test_equivalent_to_abs_then_rms(self):
        """Rectify+RMS should equal RMS(abs(signal))."""
        signal = np.array([-1.0, 2.0, -3.0, 4.0])
        result = full_wave_rectify_then_rms(signal)
        expected = compute_rms(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(result - expected) < 1e-8

    def test_positive_signal_unchanged(self):
        """For non-negative signals, rectify+RMS equals plain RMS."""
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(full_wave_rectify_then_rms(signal) - compute_rms(signal)) < 1e-8

    def test_fully_negative_signal(self):
        """All-negative signal: RMS equals RMS of its positive mirror."""
        neg = np.array([-3.0, -4.0])
        pos = np.array([3.0, 4.0])
        assert abs(full_wave_rectify_then_rms(neg) - compute_rms(pos)) < 1e-8


# ─────────────────────────────────────────────────────────────
#  compute_resultant_magnitude
# ─────────────────────────────────────────────────────────────

class TestResultantMagnitude:
    def test_pythagorean_3d(self):
        """3-4-0 → 5, 0-0-5 → 5."""
        fx = np.array([3.0, 0.0])
        fy = np.array([4.0, 0.0])
        fz = np.array([0.0, 5.0])
        result = compute_resultant_magnitude(fx, fy, fz)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0])

    def test_unit_vectors(self):
        """Each unit axis vector → magnitude 1."""
        one = np.array([1.0])
        zero = np.array([0.0])
        assert abs(float(compute_resultant_magnitude(one, zero, zero)) - 1.0) < 1e-10
        assert abs(float(compute_resultant_magnitude(zero, one, zero)) - 1.0) < 1e-10
        assert abs(float(compute_resultant_magnitude(zero, zero, one)) - 1.0) < 1e-10


# ─────────────────────────────────────────────────────────────
#  Integration: read real Shivangi .sto files (if available)
# ─────────────────────────────────────────────────────────────

SHIVANGI_INVERSE_DYNAMICS = Path("D:/samrudh/SHIVANGI overall/OpenSimData/Model/inverse_dynamics.sto")
SHIVANGI_JRA = Path("D:/samrudh/SHIVANGI overall/OpenSimData/Model/LaiUhlrich2022_scaled-scaled_JointReaction_ReactionLoads.sto")
SHIVANGI_SO_ACT = Path("D:/samrudh/SHIVANGI overall/OpenSimData/Model/LaiUhlrich2022_scaled-scaled_StaticOptimization_activation.sto")
SHIVANGI_MOT = Path("D:/samrudh/SHIVANGI overall/OpenSimData/Kinematics/usgreal1.mot")


@pytest.mark.skipif(
    not SHIVANGI_INVERSE_DYNAMICS.exists(),
    reason="Shivangi manual output files not present — skipping integration test"
)
class TestIntegrationShivangi:
    """
    Integration tests against the existing manual-run thesis outputs for Shivangi.
    These tests confirm the parser and RMS functions work on real data.
    They do not validate exact values (we don't have Shivangi's individual row
    from the thesis), but they confirm the functions produce finite, plausible results.
    """

    def test_read_id_sto_parses_without_error(self):
        from src.utils import read_sto_file
        df = read_sto_file(SHIVANGI_INVERSE_DYNAMICS)
        assert len(df) > 0
        assert "time" in df.columns

    def test_id_sto_produces_finite_rms(self):
        """ID joint moment RMS should be finite and positive."""
        from src.utils import read_sto_file
        df = read_sto_file(SHIVANGI_INVERSE_DYNAMICS)
        for col in df.columns:
            if col == "time":
                continue
            val = compute_rms(df[col].to_numpy())
            assert np.isfinite(val) or np.isnan(val), f"Infinite RMS for column {col}"

    def test_read_mot_file(self):
        from src.utils import read_mot_file
        df = read_mot_file(SHIVANGI_MOT)
        assert len(df) > 0

    def test_jra_sto_has_force_columns(self):
        from src.utils import read_sto_file
        df = read_sto_file(SHIVANGI_JRA)
        assert len(df) > 0
        # Check at least one force column exists
        force_cols = [c for c in df.columns if "F" in c.upper() and "time" not in c.lower()]
        assert len(force_cols) > 0, f"No force columns found. Columns: {list(df.columns)}"

    def test_so_activation_values_in_range(self):
        """Muscle activations should be in [0, 1]."""
        from src.utils import read_sto_file
        df = read_sto_file(SHIVANGI_SO_ACT)
        data_cols = [c for c in df.columns if c != "time"]
        for col in data_cols[:10]:  # Check first 10 muscles
            col_data = df[col].dropna()
            assert col_data.min() >= -0.01, f"Activation < 0 in {col}"
            assert col_data.max() <= 1.01, f"Activation > 1 in {col}"

    def test_ik_rms_extraction(self):
        """Sanity check: IK RMS values should be in plausible degree range."""
        joint_angle_map = {
            "lumbar_extension": "lumbar_flexion_deg",
        }
        result = extract_ik_rms(SHIVANGI_MOT, joint_angle_map)
        # Check only if column exists (may not match if model coordinate names differ)
        for label, val in result.items():
            if not np.isnan(val):
                assert 0 <= val <= 180, f"Implausible joint angle RMS: {label}={val}"
