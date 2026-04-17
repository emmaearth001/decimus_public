"""Tests for planner.py — orchestration planning, register analysis, instrument normalization."""

import pytest

from decimus.analyzer import Note
from decimus.instruments import INSTRUMENTS, get_ensemble
from decimus.planner import (
    _apply_llm_plan,
    _compute_register,
    _extract_instrument_names,
    _normalize_instrument_name,
    _notes_fit_instrument,
    _rank_by_register,
    create_plan,
)
from decimus.styles import get_style

# ---------------------------------------------------------------------------
# _compute_register
# ---------------------------------------------------------------------------

class TestComputeRegister:
    def test_empty_notes(self):
        assert _compute_register([]) == "mid"

    def test_low_register(self):
        notes = [Note(40, 0, 480), Note(45, 480, 960), Note(42, 960, 1440)]
        assert _compute_register(notes) == "low"

    def test_mid_register(self):
        notes = [Note(60, 0, 480), Note(65, 480, 960), Note(70, 960, 1440)]
        assert _compute_register(notes) == "mid"

    def test_high_register(self):
        notes = [Note(80, 0, 480), Note(85, 480, 960), Note(78, 960, 1440)]
        assert _compute_register(notes) == "high"

    def test_boundary_low(self):
        # avg=55 -> should be "mid" (not < 55)
        notes = [Note(55, 0, 480)]
        assert _compute_register(notes) == "mid"

    def test_boundary_high(self):
        # avg=75 -> should be "mid" (not > 75)
        notes = [Note(75, 0, 480)]
        assert _compute_register(notes) == "mid"


# ---------------------------------------------------------------------------
# _rank_by_register
# ---------------------------------------------------------------------------

class TestRankByRegister:
    def test_mid_register_preserves_order(self):
        names = ["violin_1", "flute", "oboe"]
        ensemble = get_ensemble("full")
        result = _rank_by_register(names, "mid", ensemble)
        assert result == names

    def test_high_register_prefers_high_instruments(self):
        names = ["cello", "flute", "trombone", "violin_1"]
        ensemble = get_ensemble("full")
        result = _rank_by_register(names, "high", ensemble)
        # flute and violin should come first
        assert result[0] in {"flute", "violin_1"}

    def test_low_register_prefers_low_instruments(self):
        names = ["violin_1", "cello", "flute", "trombone"]
        ensemble = get_ensemble("full")
        result = _rank_by_register(names, "low", ensemble)
        assert result[0] in {"cello", "trombone"}

    def test_filters_to_ensemble(self):
        names = ["violin_1", "tuba", "flute"]
        ensemble = get_ensemble("strings")  # no tuba, no flute
        result = _rank_by_register(names, "high", ensemble)
        assert "tuba" not in result
        assert "flute" not in result


# ---------------------------------------------------------------------------
# _notes_fit_instrument
# ---------------------------------------------------------------------------

class TestNotesFitInstrument:
    def test_empty_notes_always_fits(self):
        assert _notes_fit_instrument([], INSTRUMENTS["violin_1"]) is True

    def test_all_notes_in_range(self):
        notes = [Note(60, 0, 480), Note(70, 480, 960), Note(80, 960, 1440)]
        assert _notes_fit_instrument(notes, INSTRUMENTS["violin_1"]) is True

    def test_most_notes_in_range(self):
        """80% tolerance: 4/5 in range should pass."""
        spec = INSTRUMENTS["violin_1"]  # 55-105
        notes = [
            Note(60, 0, 480), Note(70, 480, 960),
            Note(80, 960, 1440), Note(90, 1440, 1920),
            Note(30, 1920, 2400),  # out of range
        ]
        assert _notes_fit_instrument(notes, spec) is True

    def test_too_many_out_of_range(self):
        """If > 20% out of range, should fail."""
        spec = INSTRUMENTS["trumpet"]  # 55-82
        notes = [
            Note(60, 0, 480),
            Note(30, 480, 960),
            Note(35, 960, 1440),
        ]
        # 1/3 in range = 33% < 80%
        assert _notes_fit_instrument(notes, spec) is False


# ---------------------------------------------------------------------------
# _normalize_instrument_name
# ---------------------------------------------------------------------------

class TestNormalizeInstrumentName:
    def test_simple_name(self):
        assert _normalize_instrument_name("violin_1") == "violin_1"

    def test_solo_prefix(self):
        assert _normalize_instrument_name("Solo Oboe") == "oboe"

    def test_with_articulation(self):
        assert _normalize_instrument_name("Solo Oboe, espressivo") == "oboe"

    def test_french_horn(self):
        assert _normalize_instrument_name("French Horn") == "horn"

    def test_double_bass(self):
        assert _normalize_instrument_name("Double Bass") == "contrabass"

    def test_muted_trumpet(self):
        assert _normalize_instrument_name("Muted Trumpet con sordino") == "trumpet"

    def test_first_violin(self):
        assert _normalize_instrument_name("First Violin") == "violin_1"

    def test_second_violin(self):
        assert _normalize_instrument_name("2nd Violin") == "violin_2"

    def test_english_horn(self):
        assert _normalize_instrument_name("English Horn") == "oboe"

    def test_unknown_returns_lowered(self):
        result = _normalize_instrument_name("theremin")
        assert result == "theremin"


# ---------------------------------------------------------------------------
# _extract_instrument_names
# ---------------------------------------------------------------------------

class TestExtractInstrumentNames:
    def test_single_instrument(self):
        result = _extract_instrument_names("Flute doubling at the octave")
        assert "flute" in result

    def test_multiple_instruments(self):
        result = _extract_instrument_names("Flute and muted horn doubling")
        assert "flute" in result
        assert "horn" in result

    def test_no_instruments(self):
        result = _extract_instrument_names("tremolo in pianissimo")
        assert result == []


# ---------------------------------------------------------------------------
# create_plan (rule-based, no LLM/KB)
# ---------------------------------------------------------------------------

class TestCreatePlan:
    def test_plan_has_melody(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.melody_tracks) >= 1

    def test_plan_has_bass(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.bass_tracks) >= 1

    def test_plan_has_harmony(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.harmony_tracks) >= 1

    def test_unique_track_ids(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        track_ids = [r.track_id for r in plan.roles]
        assert len(track_ids) == len(set(track_ids))

    def test_velocity_scales_valid(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        for role in plan.roles:
            assert 0.0 < role.velocity_scale <= 1.0

    @pytest.mark.parametrize("style", ["mozart", "beethoven", "tchaikovsky", "brahms", "mahler", "debussy", "ravel", "stravinsky", "williams", "zimmer"])
    def test_all_styles_produce_plan(self, sample_analysis, style):
        plan = create_plan(sample_analysis, style, "full",
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.roles) > 0

    @pytest.mark.parametrize("ensemble", ["full", "strings", "chamber", "winds"])
    def test_all_ensembles_produce_plan(self, sample_analysis, ensemble):
        plan = create_plan(sample_analysis, "tchaikovsky", ensemble,
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.roles) > 0

    def test_strings_only_has_strings(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "strings",
                           use_knowledge_base=False, use_llm=False)
        for role in plan.roles:
            assert role.spec.family == "strings"

    def test_plan_properties(self, sample_analysis):
        plan = create_plan(sample_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        assert plan.max_tracks == len(plan.roles)
        assert set(plan.active_tracks) == {r.track_id for r in plan.roles}

    def test_empty_analysis_still_creates_plan(self, empty_analysis):
        plan = create_plan(empty_analysis, "tchaikovsky", "full",
                           use_knowledge_base=False, use_llm=False)
        assert len(plan.roles) > 0


# ---------------------------------------------------------------------------
# _apply_llm_plan
# ---------------------------------------------------------------------------

class TestApplyLLMPlan:
    def test_valid_plan(self):
        style = get_style("tchaikovsky")
        ensemble = get_ensemble("full")
        llm_plan = {
            "melody": {"primary": "violin_1", "doublings": ["flute"]},
            "bass": {"primary": "cello", "doublings": ["contrabass"]},
            "harmony": ["viola", "clarinet"],
            "countermelody": "horn",
            "advice": "Rich romantic texture",
        }
        plan = _apply_llm_plan(llm_plan, style, ensemble, "full")
        assert plan is not None
        assert any(r.role == "melody" for r in plan.roles)
        assert any(r.role == "bass" for r in plan.roles)

    def test_missing_melody_rejected(self):
        style = get_style("tchaikovsky")
        ensemble = get_ensemble("full")
        llm_plan = {
            "bass": {"primary": "cello"},
            "harmony": ["viola"],
        }
        plan = _apply_llm_plan(llm_plan, style, ensemble, "full")
        assert plan is None

    def test_missing_bass_rejected(self):
        style = get_style("tchaikovsky")
        ensemble = get_ensemble("full")
        llm_plan = {
            "melody": {"primary": "violin_1"},
            "harmony": ["viola"],
        }
        plan = _apply_llm_plan(llm_plan, style, ensemble, "full")
        assert plan is None

    def test_unknown_instrument_skipped(self):
        style = get_style("tchaikovsky")
        ensemble = get_ensemble("full")
        llm_plan = {
            "melody": {"primary": "violin_1"},
            "bass": {"primary": "cello"},
            "harmony": ["theremin"],  # unknown
        }
        plan = _apply_llm_plan(llm_plan, style, ensemble, "full")
        assert plan is not None
        harmony = [r for r in plan.roles if r.role == "harmony"]
        assert len(harmony) == 0  # theremin skipped

    def test_mahler_format(self):
        style = get_style("mahler")
        ensemble = get_ensemble("full")
        llm_plan = {
            "primary_instrument": "Solo Oboe, espressivo",
            "doubling": "Flute doubling at the octave",
            "bass": {"primary": "cello", "doublings": ["contrabass"]},
            "harmony": ["viola"],
            "justification": "Mahler's Wunderhorn texture",
        }
        plan = _apply_llm_plan(llm_plan, style, ensemble, "full")
        assert plan is not None
        melody_roles = [r for r in plan.roles if r.role == "melody"]
        assert len(melody_roles) >= 1
