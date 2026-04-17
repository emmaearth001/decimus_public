"""Tests for orchestrator.py — note distribution, monophonic enforcement, articulations."""

from miditoolkit.midi.containers import Note as mtkNote

from decimus.analyzer import Note
from decimus.instruments import INSTRUMENTS
from decimus.orchestrator import (
    _apply_articulations,
    _compute_phrase_dynamics,
    _enforce_monophonic,
    _generate_countermelody,
    _get_active_harmony_count,
    _scale_velocity,
    orchestrate_direct,
)

# ---------------------------------------------------------------------------
# _enforce_monophonic
# ---------------------------------------------------------------------------

class TestEnforceMonophonic:
    def test_empty(self):
        assert _enforce_monophonic([]) == []

    def test_single_note(self):
        notes = [mtkNote(velocity=90, pitch=60, start=0, end=480)]
        result = _enforce_monophonic(notes)
        assert len(result) == 1

    def test_keeps_highest_at_same_start(self):
        notes = [
            mtkNote(velocity=90, pitch=60, start=0, end=480),
            mtkNote(velocity=80, pitch=72, start=0, end=480),
            mtkNote(velocity=70, pitch=55, start=0, end=480),
        ]
        result = _enforce_monophonic(notes)
        assert len(result) == 1
        assert result[0].pitch == 72

    def test_trims_overlaps(self):
        notes = [
            mtkNote(velocity=90, pitch=60, start=0, end=600),
            mtkNote(velocity=80, pitch=64, start=480, end=960),
        ]
        result = _enforce_monophonic(notes)
        assert len(result) == 2
        assert result[0].end == 480  # trimmed to next note's start

    def test_non_overlapping_unchanged(self):
        notes = [
            mtkNote(velocity=90, pitch=60, start=0, end=480),
            mtkNote(velocity=80, pitch=64, start=480, end=960),
        ]
        result = _enforce_monophonic(notes)
        assert result[0].end == 480
        assert result[1].start == 480


# ---------------------------------------------------------------------------
# _scale_velocity
# ---------------------------------------------------------------------------

class TestScaleVelocity:
    def test_normal_scale(self):
        assert _scale_velocity(100, 0.8) == 80

    def test_clamp_low(self):
        assert _scale_velocity(1, 0.1) == 1  # min is 1

    def test_clamp_high(self):
        assert _scale_velocity(127, 1.5) == 127  # max is 127

    def test_zero_velocity_becomes_one(self):
        assert _scale_velocity(0, 1.0) == 1

    def test_scale_one(self):
        assert _scale_velocity(90, 1.0) == 90


# ---------------------------------------------------------------------------
# _apply_articulations
# ---------------------------------------------------------------------------

class TestApplyArticulations:
    def test_single_note_no_crash(self):
        """Should not crash with fewer than 2 notes."""
        notes = [mtkNote(velocity=90, pitch=60, start=0, end=480)]
        _apply_articulations(notes, 480)  # should not raise

    def test_staccato_shortens(self):
        """Short notes with gaps should be shortened."""
        tpb = 480
        notes = [
            mtkNote(velocity=80, pitch=60, start=0, end=tpb // 4),       # short note
            mtkNote(velocity=80, pitch=62, start=tpb, end=tpb + tpb // 4),  # next note with gap
        ]
        original_dur = notes[0].end - notes[0].start
        _apply_articulations(notes, tpb)
        new_dur = notes[0].end - notes[0].start
        assert new_dur <= original_dur

    def test_accent_boosts_velocity(self):
        """A loud note surrounded by quiet ones should get a velocity boost."""
        tpb = 480
        notes = [
            mtkNote(velocity=50, pitch=60, start=0, end=tpb),
            mtkNote(velocity=50, pitch=62, start=tpb, end=2*tpb),
            mtkNote(velocity=100, pitch=64, start=2*tpb, end=3*tpb),  # accent
            mtkNote(velocity=50, pitch=65, start=3*tpb, end=4*tpb),
            mtkNote(velocity=50, pitch=67, start=4*tpb, end=5*tpb),
        ]
        _apply_articulations(notes, tpb)
        # The accent note (index 2) should have boosted velocity
        assert notes[2].velocity >= 100


# ---------------------------------------------------------------------------
# _compute_phrase_dynamics
# ---------------------------------------------------------------------------

class TestComputePhraseDynamics:
    def test_empty_analysis(self, empty_analysis):
        result = _compute_phrase_dynamics(empty_analysis, 480)
        assert result == []

    def test_single_phrase(self, sample_analysis):
        result = _compute_phrase_dynamics(sample_analysis, 480)
        assert len(result) >= 1
        for start, end, ratio in result:
            assert start < end
            assert 0.0 <= ratio <= 1.0

    def test_multiple_phrases(self, sample_analysis):
        # Add more phrase boundaries
        sample_analysis.phrase_boundaries = [0, 1]
        result = _compute_phrase_dynamics(sample_analysis, 480)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _get_active_harmony_count
# ---------------------------------------------------------------------------

class TestGetActiveHarmonyCount:
    def test_no_dynamics_returns_total(self):
        assert _get_active_harmony_count(0, [], 5) == 5

    def test_loud_phrase_uses_all(self):
        dynamics = [(0, 1000, 1.0)]  # max dynamic
        result = _get_active_harmony_count(500, dynamics, 10)
        assert result == 10

    def test_quiet_phrase_uses_fewer(self):
        dynamics = [(0, 1000, 0.0)]  # min dynamic
        result = _get_active_harmony_count(500, dynamics, 10)
        assert result < 10
        assert result >= 2  # min is 2

    def test_outside_range_returns_total(self):
        dynamics = [(0, 500, 0.5)]
        result = _get_active_harmony_count(600, dynamics, 8)
        assert result == 8


# ---------------------------------------------------------------------------
# _generate_countermelody
# ---------------------------------------------------------------------------

class TestGenerateCountermelody:
    def test_too_few_notes_returns_empty(self):
        notes = [Note(60, 0, 480, 90), Note(62, 480, 960, 90)]
        result = _generate_countermelody(notes, INSTRUMENTS["cello"])
        assert result == []

    def test_generates_notes(self, melody_notes):
        result = _generate_countermelody(melody_notes, INSTRUMENTS["cello"])
        assert len(result) > 0

    def test_countermelody_in_range(self, melody_notes):
        spec = INSTRUMENTS["cello"]
        result = _generate_countermelody(melody_notes, spec)
        for note in result:
            assert spec.low <= note.pitch <= spec.high

    def test_countermelody_lower_velocity(self, melody_notes):
        result = _generate_countermelody(melody_notes, INSTRUMENTS["cello"])
        for note in result:
            assert note.velocity < 90  # should be 70% of original

    def test_countermelody_skips_notes(self, melody_notes):
        """Counter should skip every other note for longer rhythm."""
        result = _generate_countermelody(melody_notes, INSTRUMENTS["cello"])
        # Should have fewer notes than melody (skips every other)
        assert len(result) < len(melody_notes)


# ---------------------------------------------------------------------------
# orchestrate_direct (integration)
# ---------------------------------------------------------------------------

class TestOrchestrateDirectIntegration:
    def test_produces_output(self, sample_analysis, romantic_plan, tmp_midi_path):
        result = orchestrate_direct(sample_analysis, romantic_plan, tmp_midi_path)
        assert result["total_notes"] > 0
        assert result["num_tracks"] > 0
        assert result["output_path"] == tmp_midi_path

    def test_output_file_created(self, sample_analysis, romantic_plan, tmp_midi_path):
        import os
        orchestrate_direct(sample_analysis, romantic_plan, tmp_midi_path)
        assert os.path.exists(tmp_midi_path)

    def test_empty_analysis_produces_empty(self, empty_analysis, tmp_midi_path):
        from decimus.planner import create_plan
        plan = create_plan(empty_analysis, style_name="mozart", use_knowledge_base=False, use_llm=False)
        result = orchestrate_direct(empty_analysis, plan, tmp_midi_path)
        # May have countermelody or not, but should not crash
        assert isinstance(result, dict)

    def test_strings_only(self, sample_analysis, strings_plan, tmp_midi_path):
        result = orchestrate_direct(sample_analysis, strings_plan, tmp_midi_path)
        assert result["total_notes"] > 0

    def test_track_names_in_result(self, sample_analysis, romantic_plan, tmp_midi_path):
        result = orchestrate_direct(sample_analysis, romantic_plan, tmp_midi_path)
        # Should have display names like "Violin I", "Cello", etc.
        for track_name in result["tracks"]:
            assert isinstance(track_name, str)
            assert len(track_name) > 0
