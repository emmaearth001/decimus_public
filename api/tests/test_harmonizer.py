"""Tests for harmonizer.py — chord progressions, voicing, bass MIDI conversion."""

from decimus.harmonizer import (
    MAJOR_CHORD_MAP,
    MAJOR_PROGRESSIONS,
    MINOR_CHORD_MAP,
    MINOR_PROGRESSIONS,
    ChordEvent,
    HarmonyNote,
    _name_to_bass_midi,
)

# ---------------------------------------------------------------------------
# _name_to_bass_midi
# ---------------------------------------------------------------------------

class TestNameToBassMidi:
    def test_c2(self):
        # C at octave 2 = (2+1)*12 + 0 = 36
        assert _name_to_bass_midi("C", 2) == 36

    def test_g2(self):
        # G at octave 2 = 36 + 7 = 43
        assert _name_to_bass_midi("G", 2) == 43

    def test_a3(self):
        # A at octave 3 = (3+1)*12 + 9 = 57
        assert _name_to_bass_midi("A", 3) == 57

    def test_sharp(self):
        assert _name_to_bass_midi("F#", 2) == 42

    def test_flat(self):
        # E- (Eb) at octave 2 = 36 + 3 = 39
        assert _name_to_bass_midi("E-", 2) == 39

    def test_unknown_defaults_to_c(self):
        result = _name_to_bass_midi("X", 2)
        assert result == 36  # defaults to pc=0 (C)


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestChordMaps:
    def test_minor_chord_map_all_degrees(self):
        for degree in range(1, 8):
            assert degree in MINOR_CHORD_MAP
            assert len(MINOR_CHORD_MAP[degree]) > 0

    def test_major_chord_map_all_degrees(self):
        for degree in range(1, 8):
            assert degree in MAJOR_CHORD_MAP
            assert len(MAJOR_CHORD_MAP[degree]) > 0

    def test_chord_map_entries_have_priority(self):
        for degree, entries in MINOR_CHORD_MAP.items():
            for numeral, priority in entries:
                assert isinstance(numeral, str)
                assert isinstance(priority, int)
                assert priority > 0


class TestProgressions:
    def test_minor_progressions_complete(self):
        """All minor chord map numerals should have progression rules."""
        all_numerals = set()
        for entries in MINOR_CHORD_MAP.values():
            for numeral, _ in entries:
                all_numerals.add(numeral)
        for numeral in all_numerals:
            assert numeral in MINOR_PROGRESSIONS, f"{numeral} missing from MINOR_PROGRESSIONS"

    def test_major_progressions_complete(self):
        all_numerals = set()
        for entries in MAJOR_CHORD_MAP.values():
            for numeral, _ in entries:
                all_numerals.add(numeral)
        for numeral in all_numerals:
            assert numeral in MAJOR_PROGRESSIONS, f"{numeral} missing from MAJOR_PROGRESSIONS"

    def test_progressions_targets_exist(self):
        """All progression targets should themselves be keys in the map."""
        for src, targets in MINOR_PROGRESSIONS.items():
            for target in targets:
                assert target in MINOR_PROGRESSIONS, \
                    f"{src} -> {target}, but {target} not in MINOR_PROGRESSIONS"


class TestHarmonyNote:
    def test_defaults(self):
        hn = HarmonyNote(pitch=60, start=0, end=480, voice="alto")
        assert hn.velocity == 80

    def test_voice_values(self):
        for voice in ["soprano", "alto", "tenor", "bass"]:
            hn = HarmonyNote(pitch=60, start=0, end=480, voice=voice)
            assert hn.voice == voice


class TestChordEvent:
    def test_creation(self):
        ce = ChordEvent(
            numeral="i", root="G", pitches=["G", "Bb", "D"],
            beat=0.0, duration=2.0, bass_note=43,
        )
        assert ce.numeral == "i"
        assert ce.bass_note == 43
        assert len(ce.pitches) == 3
