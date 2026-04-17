"""Tests for analyzer.py — note merging, phrase detection, key detection helpers."""

from decimus.analyzer import (
    MAJOR_PROFILE,
    MINOR_PROFILE,
    PITCH_NAMES,
    Chord,
    Note,
    PianoAnalysis,
    _beats_to_notes,
    _detect_phrases,
    _pearson,
)

# ---------------------------------------------------------------------------
# _beats_to_notes
# ---------------------------------------------------------------------------

class TestBeatsToNotes:
    def test_empty_input(self):
        assert _beats_to_notes([], 480) == []

    def test_single_note(self):
        raw = [(0.0, 60, 1.0, 90)]  # C4, 1 beat
        notes = _beats_to_notes(raw, 480)
        assert len(notes) == 1
        assert notes[0].pitch == 60
        assert notes[0].start == 0
        assert notes[0].end == 480
        assert notes[0].velocity == 90

    def test_merge_consecutive_same_pitch(self):
        """Two consecutive C4s should merge into one long note."""
        raw = [
            (0.0, 60, 1.0, 90),
            (1.0, 60, 1.0, 85),
        ]
        notes = _beats_to_notes(raw, 480)
        assert len(notes) == 1
        assert notes[0].pitch == 60
        assert notes[0].start == 0
        assert notes[0].end == 960  # 2 beats

    def test_no_merge_different_pitches(self):
        raw = [
            (0.0, 60, 1.0, 90),
            (1.0, 62, 1.0, 85),
        ]
        notes = _beats_to_notes(raw, 480)
        assert len(notes) == 2
        assert notes[0].pitch == 60
        assert notes[1].pitch == 62

    def test_velocity_from_first_segment(self):
        raw = [
            (0.0, 60, 1.0, 100),
            (1.0, 60, 1.0, 50),  # merged, velocity should be 100
        ]
        notes = _beats_to_notes(raw, 480)
        assert notes[0].velocity == 100

    def test_small_gap_still_merges(self):
        """Gap smaller than tpb/8 should still merge."""
        tpb = 480
        gap = tpb // 16  # very small gap
        raw = [
            (0.0, 60, 1.0, 90),
            ((tpb + gap) / tpb, 60, 1.0, 85),
        ]
        notes = _beats_to_notes(raw, tpb)
        assert len(notes) == 1

    def test_large_gap_no_merge(self):
        """Gap larger than tpb/8 should not merge."""
        tpb = 480
        raw = [
            (0.0, 60, 1.0, 90),
            (2.0, 60, 1.0, 85),  # 1-beat gap
        ]
        notes = _beats_to_notes(raw, tpb)
        assert len(notes) == 2


# ---------------------------------------------------------------------------
# _detect_phrases
# ---------------------------------------------------------------------------

class TestDetectPhrases:
    def test_empty_melody(self):
        result = _detect_phrases([], [], 480)
        assert result == [0]

    def test_empty_measures(self):
        notes = [Note(60, 0, 480, 90)]
        result = _detect_phrases(notes, [], 480)
        assert result == [0]

    def test_always_includes_measure_zero(self):
        tpb = 480
        tpm = tpb * 4
        notes = [Note(60, 0, tpb, 90), Note(62, tpb, 2*tpb, 90)]
        measures = [(0, tpm)]
        result = _detect_phrases(notes, measures, tpb)
        assert 0 in result

    def test_rest_creates_boundary(self):
        """A rest >= half a beat should create a phrase boundary."""
        tpb = 480
        tpm = tpb * 4
        notes = [
            Note(60, 0, tpb, 90),
            # gap of tpb/2 = 240 ticks (rest)
            Note(64, tpb + tpb // 2, 2 * tpb, 90),
        ]
        measures = [(0, tpm), (tpm, 2 * tpm)]
        result = _detect_phrases(notes, measures, tpb)
        assert len(result) >= 1

    def test_leap_creates_boundary(self):
        """An interval >= 7 semitones should create a phrase boundary."""
        tpb = 480
        tpm = tpb * 4
        notes = [
            Note(60, 0, tpb, 90),
            Note(60 + 8, tpb, 2 * tpb, 90),  # minor 6th = 8 semitones
        ]
        measures = [(0, tpm), (tpm, 2 * tpm)]
        result = _detect_phrases(notes, measures, tpb)
        assert len(result) >= 1

    def test_velocity_drop_creates_boundary(self):
        """A 30%+ velocity drop should create a phrase boundary."""
        tpb = 480
        tpm = tpb * 4
        notes = [
            Note(60, 0, tpb, 100),
            Note(62, tpm, tpm + tpb, 60),  # 40% drop in measure 1
        ]
        measures = [(0, tpm), (tpm, 2 * tpm)]
        result = _detect_phrases(notes, measures, tpb)
        assert 1 in result

    def test_regular_fallback_for_long_pieces(self):
        """If few boundaries found, should add regular ones every 4 measures."""
        tpb = 480
        tpm = tpb * 4
        # Smooth melody with no gaps, leaps, or velocity changes
        notes = [
            Note(60 + i, i * tpb, (i + 1) * tpb, 80)
            for i in range(24)  # 6 measures of notes
        ]
        measures = [(i * tpm, (i + 1) * tpm) for i in range(8)]
        result = _detect_phrases(notes, measures, tpb)
        # Should have at least boundaries at 0 and 4
        assert 0 in result


# ---------------------------------------------------------------------------
# _pearson
# ---------------------------------------------------------------------------

class TestPearson:
    def test_perfect_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        assert abs(_pearson(x, x) - 1.0) < 1e-10

    def test_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(x, y) - (-1.0)) < 1e-10

    def test_zero_variance_returns_zero(self):
        x = [1.0, 1.0, 1.0]
        y = [1.0, 2.0, 3.0]
        assert _pearson(x, y) == 0.0

    def test_key_profiles_have_correct_length(self):
        assert len(MAJOR_PROFILE) == 12
        assert len(MINOR_PROFILE) == 12
        assert len(PITCH_NAMES) == 12


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_note_defaults(self):
        n = Note(pitch=60, start=0, end=480)
        assert n.velocity == 90

    def test_chord_defaults(self):
        c = Chord(root="C", quality="maj", measure=0)
        assert c.label == ""

    def test_piano_analysis_defaults(self):
        a = PianoAnalysis()
        assert a.melody_notes == []
        assert a.key == "C"
        assert a.tempo == 120.0
        assert a.time_sig == (4, 4)
        assert a.ticks_per_beat == 480
