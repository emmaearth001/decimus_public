"""Build a multi-track prime MIDI from piano analysis and orchestration plan.

The core strategy: distribute the piano content across orchestral instruments
to create a synthetic multi-track MIDI, then feed it to SymphonyNet's
process_prime_midi() as conditioning context.
"""

import os
import tempfile

from miditoolkit.midi.containers import Instrument
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile

from .analyzer import Note, PianoAnalysis
from .instruments import clamp_to_range
from .planner import InstrumentRole, OrchestrationPlan


def build_prime_midi(
    analysis: PianoAnalysis,
    plan: OrchestrationPlan,
    prime_measures: int = 4,
) -> str:
    """Build a multi-track prime MIDI and return path to temp file.

    Distributes the first `prime_measures` of analyzed piano content
    across the orchestral instruments specified in the plan.
    """
    tpb = 480  # ticks per beat, matching SymphonyNet's default
    midi = MidiFile(ticks_per_beat=tpb)

    # Calculate measure boundaries in ticks
    beats_per_mea = analysis.time_sig[0] * (4 / analysis.time_sig[1])
    ticks_per_mea = int(tpb * beats_per_mea)
    prime_end_tick = prime_measures * ticks_per_mea

    # Group notes by role for the prime region
    melody_notes = [n for n in analysis.melody_notes if n.start < prime_end_tick]
    bass_notes = [n for n in analysis.bass_notes if n.start < prime_end_tick]
    inner_notes = [n for n in analysis.inner_notes if n.start < prime_end_tick]

    # Create one Instrument per role in the plan
    instruments_by_track: dict[int, Instrument] = {}

    for role in plan.roles:
        inst = Instrument(
            program=role.spec.program % 128,
            is_drum=role.spec.is_drum,
            name=role.spec.display_name,
        )
        instruments_by_track[role.track_id] = inst

    # Distribute notes to instruments based on their roles
    for role in plan.roles:
        inst = instruments_by_track[role.track_id]

        if role.role in ("melody", "doubling"):
            _assign_notes(inst, melody_notes, role)
        elif role.role == "bass":
            _assign_notes(inst, bass_notes, role)
        elif role.role == "countermelody":
            # Use inner voice notes for countermelody
            _assign_notes(inst, inner_notes, role)
        elif role.role == "harmony":
            # Distribute inner notes across harmony instruments
            _assign_notes(inst, inner_notes, role)

    # Add all non-empty instruments to MIDI
    for inst in instruments_by_track.values():
        if inst.notes:
            midi.instruments.append(inst)

    # Save to temp file
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "decimus_prime.mid")
    midi.dump(tmp_path)
    return tmp_path


def _assign_notes(
    instrument: Instrument,
    source_notes: list[Note],
    role: InstrumentRole,
) -> None:
    """Assign source notes to an instrument, clamping to its playable range."""
    for note in source_notes:
        pitch = clamp_to_range(note.pitch, role.spec)

        # Skip if still out of range after clamping
        if pitch < role.spec.low or pitch > role.spec.high:
            continue

        velocity = int(90 * role.velocity_scale)
        instrument.notes.append(mtkNote(
            velocity=velocity,
            pitch=pitch,
            start=note.start,
            end=note.end,
        ))
