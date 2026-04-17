"""Output writing: MIDI with proper track names and instrument programs."""

import os

from miditoolkit.midi.containers import Instrument, TempoChange, TimeSignature
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile

from .planner import OrchestrationPlan


def write_orchestral_midi(
    note_seq: list[tuple],
    plan: OrchestrationPlan,
    output_path: str,
    ticks_per_beat: int = 480,
    tempo: float = 120.0,
    time_sig: tuple[int, int] = (4, 4),
) -> str:
    """Write an orchestral MIDI file with proper track names and programs.

    note_seq: list of (pitch, program, start, end, track_id) or
              (pitch, program, start, end, track_id, velocity)
    """
    ticks_per_32nd = ticks_per_beat // 8
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    # Write tempo and time signature
    midi.tempo_changes = [TempoChange(tempo=tempo, time=0)]
    midi.time_signature_changes = [
        TimeSignature(numerator=time_sig[0], denominator=time_sig[1], time=0)
    ]

    # Group notes by track
    tracks: dict[int, list] = {}
    for note in note_seq:
        track_id = note[4]
        tracks.setdefault(track_id, []).append(note)

    # Create instruments ordered by role
    for role in plan.roles:
        tid = role.track_id
        notes = tracks.get(tid, [])
        if not notes:
            continue

        inst = Instrument(
            program=role.spec.program % 128,
            is_drum=role.spec.is_drum,
            name=role.spec.display_name,
        )

        for note in notes:
            velocity = note[5] if len(note) > 5 else 90
            pitch = note[0]
            start = note[2] * ticks_per_32nd
            end = note[3] * ticks_per_32nd

            inst.notes.append(mtkNote(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            ))

        inst.remove_invalid_notes(verbose=False)
        if inst.notes:
            midi.instruments.append(inst)

    # Also write any tracks not in the plan (model may have generated extras)
    for tid, notes in tracks.items():
        if tid in plan.track_to_instrument:
            continue
        if not notes:
            continue
        # Use the program from the note data
        program = notes[0][1] % 128
        inst = Instrument(program=program, name=f"Track {tid}")
        for note in notes:
            velocity = note[5] if len(note) > 5 else 90
            inst.notes.append(mtkNote(velocity, note[0], note[2] * ticks_per_32nd, note[3] * ticks_per_32nd))
        inst.remove_invalid_notes(verbose=False)
        if inst.notes:
            midi.instruments.append(inst)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    midi.dump(output_path)
    return output_path


def summarize_output(note_seq: list[tuple], plan: OrchestrationPlan) -> dict:
    """Generate a summary of the orchestral output."""
    track_counts: dict[str, int] = {}
    total_notes = len(note_seq)

    for role in plan.roles:
        tid = role.track_id
        count = sum(1 for n in note_seq if n[4] == tid)
        if count > 0:
            track_counts[role.spec.display_name] = count

    # Estimate duration from note positions
    if note_seq:
        max_end = max(n[3] for n in note_seq)
        # 32nd notes to seconds: 32nd = (60 / tempo / 8)
        duration_32nds = max_end
    else:
        duration_32nds = 0

    return {
        "total_notes": total_notes,
        "tracks": track_counts,
        "num_tracks": len(track_counts),
        "duration_32nds": duration_32nds,
    }
