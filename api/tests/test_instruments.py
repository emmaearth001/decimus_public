"""Tests for instruments.py — instrument database, ranges, and ensembles."""

import pytest

from decimus.instruments import (
    ENSEMBLES,
    INSTRUMENTS,
    clamp_to_range,
    get_ensemble,
    in_range,
)

# ---------------------------------------------------------------------------
# InstrumentSpec validation
# ---------------------------------------------------------------------------

class TestInstrumentSpecs:
    def test_all_instruments_have_valid_ranges(self):
        for name, spec in INSTRUMENTS.items():
            assert spec.low < spec.high, f"{name}: low ({spec.low}) >= high ({spec.high})"
            assert 0 <= spec.low <= 127, f"{name}: low out of MIDI range"
            assert 0 <= spec.high <= 127, f"{name}: high out of MIDI range"

    def test_all_instruments_have_valid_programs(self):
        for name, spec in INSTRUMENTS.items():
            assert 0 <= spec.program <= 127, f"{name}: program {spec.program} out of GM range"

    def test_all_instruments_have_family(self):
        valid_families = {"strings", "woodwinds", "brass", "percussion", "keyboard"}
        for name, spec in INSTRUMENTS.items():
            assert spec.family in valid_families, f"{name}: unknown family {spec.family!r}"

    def test_woodwinds_are_monophonic(self):
        for name, spec in INSTRUMENTS.items():
            if spec.family == "woodwinds":
                assert spec.is_monophonic, f"{name} is woodwind but not monophonic"

    def test_brass_are_monophonic(self):
        for name, spec in INSTRUMENTS.items():
            if spec.family == "brass":
                assert spec.is_monophonic, f"{name} is brass but not monophonic"

    def test_strings_are_polyphonic(self):
        """Strings (except harp) should not be monophonic."""
        for name in ["violin_1", "violin_2", "viola", "cello", "contrabass"]:
            assert not INSTRUMENTS[name].is_monophonic

    def test_instrument_count(self):
        assert len(INSTRUMENTS) >= 15

    def test_frozen_dataclass(self):
        spec = INSTRUMENTS["violin_1"]
        with pytest.raises(AttributeError):
            spec.name = "modified"


# ---------------------------------------------------------------------------
# get_ensemble
# ---------------------------------------------------------------------------

class TestGetEnsemble:
    def test_full_ensemble(self):
        ensemble = get_ensemble("full")
        assert len(ensemble) == 16
        names = {s.name for s in ensemble}
        assert "violin_1" in names
        assert "tuba" in names
        assert "harp" in names

    def test_strings_ensemble(self):
        ensemble = get_ensemble("strings")
        assert len(ensemble) == 5
        families = {s.family for s in ensemble}
        assert families == {"strings"}

    def test_chamber_ensemble(self):
        ensemble = get_ensemble("chamber")
        assert len(ensemble) == 10

    def test_winds_ensemble(self):
        ensemble = get_ensemble("winds")
        assert len(ensemble) == 7
        for spec in ensemble:
            assert spec.family in {"woodwinds", "brass"}

    def test_unknown_ensemble_raises(self):
        with pytest.raises(ValueError, match="Unknown ensemble"):
            get_ensemble("jazz")

    def test_ensemble_no_duplicates(self):
        for ens_name in ENSEMBLES:
            ensemble = get_ensemble(ens_name)
            names = [s.name for s in ensemble]
            assert len(names) == len(set(names)), f"Duplicates in {ens_name}"

    def test_all_ensemble_instruments_exist(self):
        for ens_name, keys in ENSEMBLES.items():
            for key in keys:
                assert key in INSTRUMENTS, f"{key} in {ens_name} not in INSTRUMENTS"


# ---------------------------------------------------------------------------
# clamp_to_range
# ---------------------------------------------------------------------------

class TestClampToRange:
    def test_pitch_in_range_unchanged(self):
        spec = INSTRUMENTS["violin_1"]  # 55-105
        assert clamp_to_range(60, spec) == 60

    def test_pitch_below_range_transposed_up(self):
        spec = INSTRUMENTS["violin_1"]  # 55-105
        result = clamp_to_range(43, spec)  # G2 -> should go up
        assert spec.low <= result <= spec.high

    def test_pitch_above_range_transposed_down(self):
        spec = INSTRUMENTS["tuba"]  # 24-60
        result = clamp_to_range(72, spec)  # C5 -> should go down
        assert spec.low <= result <= spec.high

    def test_clamp_preserves_pitch_class(self):
        spec = INSTRUMENTS["violin_1"]
        original = 43  # G2
        result = clamp_to_range(original, spec)
        assert result % 12 == original % 12  # same pitch class

    def test_clamp_at_boundary(self):
        spec = INSTRUMENTS["violin_1"]  # 55-105
        assert clamp_to_range(55, spec) == 55  # exact low
        assert clamp_to_range(105, spec) == 105  # exact high

    def test_clamp_impossible_pitch(self):
        """If octave transposition can't fix it, return closest possible."""
        spec = INSTRUMENTS["timpani"]  # 40-55, narrow range
        result = clamp_to_range(30, spec)
        # Should try +12 = 42, which is in range
        assert result == 42


# ---------------------------------------------------------------------------
# in_range
# ---------------------------------------------------------------------------

class TestInRange:
    def test_in_range_true(self):
        spec = INSTRUMENTS["violin_1"]
        assert in_range(60, spec) is True
        assert in_range(55, spec) is True   # low boundary
        assert in_range(105, spec) is True  # high boundary

    def test_in_range_false(self):
        spec = INSTRUMENTS["violin_1"]  # 55-105
        assert in_range(54, spec) is False
        assert in_range(106, spec) is False

    def test_in_range_extreme(self):
        assert in_range(0, INSTRUMENTS["piano"]) is False
        assert in_range(21, INSTRUMENTS["piano"]) is True
        assert in_range(108, INSTRUMENTS["piano"]) is True
        assert in_range(127, INSTRUMENTS["piano"]) is False
