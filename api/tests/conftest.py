"""Shared fixtures for Decimus test suite."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decimus.analyzer import Chord, Note, PianoAnalysis

# ---------------------------------------------------------------------------
# Note / Analysis Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_note():
    """A single middle-C note."""
    return Note(pitch=60, start=0, end=480, velocity=90)


@pytest.fixture
def melody_notes():
    """8-note G minor melody (measures 1-2)."""
    tpb = 480
    return [
        Note(pitch=67, start=0,        end=tpb,     velocity=90),   # G4
        Note(pitch=70, start=tpb,      end=2*tpb,   velocity=85),   # Bb4
        Note(pitch=72, start=2*tpb,    end=3*tpb,   velocity=95),   # C5
        Note(pitch=74, start=3*tpb,    end=4*tpb,   velocity=80),   # D5
        Note(pitch=72, start=4*tpb,    end=5*tpb,   velocity=88),   # C5
        Note(pitch=70, start=5*tpb,    end=6*tpb,   velocity=82),   # Bb4
        Note(pitch=67, start=6*tpb,    end=7*tpb,   velocity=90),   # G4
        Note(pitch=65, start=7*tpb,    end=8*tpb,   velocity=75),   # F4
    ]


@pytest.fixture
def bass_notes():
    """Simple bass line in G minor."""
    tpb = 480
    return [
        Note(pitch=43, start=0,       end=2*tpb, velocity=80),   # G2
        Note(pitch=48, start=2*tpb,   end=4*tpb, velocity=75),   # C3
        Note(pitch=50, start=4*tpb,   end=6*tpb, velocity=78),   # D3
        Note(pitch=43, start=6*tpb,   end=8*tpb, velocity=80),   # G2
    ]


@pytest.fixture
def inner_notes():
    """Simple inner voice notes."""
    tpb = 480
    return [
        Note(pitch=58, start=0,       end=2*tpb, velocity=65),   # Bb3
        Note(pitch=60, start=2*tpb,   end=4*tpb, velocity=60),   # C4
        Note(pitch=62, start=4*tpb,   end=6*tpb, velocity=63),   # D4
        Note(pitch=58, start=6*tpb,   end=8*tpb, velocity=65),   # Bb3
    ]


@pytest.fixture
def sample_analysis(melody_notes, bass_notes, inner_notes):
    """A complete PianoAnalysis for G minor, 2 measures."""
    tpb = 480
    tpm = tpb * 4  # ticks per measure (4/4)
    return PianoAnalysis(
        melody_notes=melody_notes,
        bass_notes=bass_notes,
        inner_notes=inner_notes,
        chords=[
            Chord(root="G", quality="minor", measure=0, label="Gm"),
            Chord(root="C", quality="minor", measure=1, label="Cm"),
        ],
        measures=[(0, tpm), (tpm, 2*tpm)],
        key="Gm",
        tempo=85.0,
        time_sig=(4, 4),
        total_measures=2,
        phrase_boundaries=[0],
        ticks_per_beat=tpb,
    )


@pytest.fixture
def empty_analysis():
    """An empty PianoAnalysis (no notes)."""
    return PianoAnalysis()


@pytest.fixture
def single_note_analysis(single_note):
    """Analysis with a single note as melody."""
    tpb = 480
    tpm = tpb * 4
    return PianoAnalysis(
        melody_notes=[single_note],
        bass_notes=[],
        inner_notes=[],
        chords=[],
        measures=[(0, tpm)],
        key="C",
        tempo=120.0,
        time_sig=(4, 4),
        total_measures=1,
        phrase_boundaries=[0],
        ticks_per_beat=tpb,
    )


# ---------------------------------------------------------------------------
# Plan Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def romantic_plan(sample_analysis):
    """A rule-based tchaikovsky/full orchestration plan."""
    from decimus.planner import create_plan
    return create_plan(
        sample_analysis,
        style_name="tchaikovsky",
        ensemble_name="full",
        use_knowledge_base=False,
        use_llm=False,
    )


@pytest.fixture
def strings_plan(sample_analysis):
    """A rule-based strings-only plan."""
    from decimus.planner import create_plan
    return create_plan(
        sample_analysis,
        style_name="mozart",
        ensemble_name="strings",
        use_knowledge_base=False,
        use_llm=False,
    )


# ---------------------------------------------------------------------------
# Temp file helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_midi_path(tmp_path):
    """Return a temp path for MIDI output."""
    return str(tmp_path / "test_output.mid")
