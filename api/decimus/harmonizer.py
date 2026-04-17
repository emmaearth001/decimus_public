"""Re-harmonizer: generates new harmonization for an extracted melody.

Takes a melody line and creates a full string arrangement with:
- New chord progression based on the key and melody
- Proper voice leading (SATB-style)
- Bass line following Rimsky-Korsakov principles
- Inner voices distributed across string section
"""

import os
from dataclasses import dataclass

from miditoolkit.midi.containers import Instrument, TempoChange
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile

from .analyzer import Note, PianoAnalysis


@dataclass
class HarmonyNote:
    """A note in the harmonization with its voice assignment."""
    pitch: int
    start: int   # ticks
    end: int
    voice: str   # "soprano", "alto", "tenor", "bass"
    velocity: int = 80


@dataclass
class ChordEvent:
    """A chord at a specific time position."""
    numeral: str      # e.g. "i", "V7", "VI"
    root: str         # e.g. "G", "D", "Eb"
    pitches: list[str]  # e.g. ["G", "Bb", "D"]
    beat: float
    duration: float   # in beats
    bass_note: int    # MIDI pitch for bass


# Chord progression templates for minor keys, indexed by melody scale degree
# Each maps a melody note (scale degree 1-7) to preferred chord choices
MINOR_CHORD_MAP = {
    # scale degree -> list of (roman numeral, priority)
    1: [("i", 3), ("VI", 2), ("iv", 1)],          # G -> Gm, Eb, Cm
    2: [("V", 3), ("V7", 3), ("VII", 2), ("iio", 1)],  # A -> D, D7, F, Adim
    3: [("i", 3), ("III", 3), ("VI", 2)],          # Bb -> Gm, Bb, Eb
    4: [("iv", 3), ("iio", 2), ("VII", 1)],        # C -> Cm, Adim, F
    5: [("i", 3), ("V", 2), ("III", 2)],           # D -> Gm, D, Bb
    6: [("VI", 3), ("iv", 2), ("iio", 1)],         # Eb -> Eb, Cm, Adim
    7: [("VII", 3), ("V", 2), ("V7", 2)],          # F -> F, D, D7
}

# Chord progression templates for major keys
MAJOR_CHORD_MAP = {
    # scale degree -> list of (roman numeral, priority)
    1: [("I", 3), ("vi", 2), ("IV", 1)],
    2: [("V", 3), ("V7", 3), ("ii", 2), ("viio", 1)],
    3: [("I", 3), ("iii", 2), ("vi", 2)],
    4: [("IV", 3), ("ii", 2), ("I", 1)],
    5: [("I", 3), ("V", 2), ("iii", 1)],
    6: [("vi", 3), ("IV", 2), ("ii", 1)],
    7: [("V", 3), ("V7", 3), ("viio", 2)],
}

# Voice leading rules: preferred chord transitions (minor keys)
MINOR_PROGRESSIONS = {
    "i": ["iv", "V", "V7", "III", "VI", "VII", "iio"],
    "iio": ["V", "V7", "i"],
    "III": ["iv", "VI", "VII", "i"],
    "iv": ["V", "V7", "i", "VII", "iio"],
    "V": ["i", "VI"],
    "V7": ["i", "VI"],
    "VI": ["iv", "iio", "V", "V7", "III", "VII"],
    "VII": ["i", "III", "V", "V7"],
    "viio": ["i", "V"],
}

# Voice leading rules: preferred chord transitions (major keys)
MAJOR_PROGRESSIONS = {
    "I": ["IV", "V", "V7", "vi", "ii", "iii"],
    "ii": ["V", "V7", "viio"],
    "iii": ["vi", "IV", "ii"],
    "IV": ["V", "V7", "I", "ii", "viio"],
    "V": ["I", "vi"],
    "V7": ["I", "vi"],
    "vi": ["IV", "ii", "V", "V7", "iii"],
    "viio": ["I", "iii"],
}

# Combined for backward compatibility
GOOD_PROGRESSIONS = {**MINOR_PROGRESSIONS, **MAJOR_PROGRESSIONS}


def harmonize_melody(
    analysis: PianoAnalysis,
    style: str = "romantic",
    output_path: str | None = None,
) -> dict:
    """Re-harmonize the melody from a piano analysis.

    Creates a new string arrangement with fresh chord voicings.

    Returns dict with output path and summary info.
    """
    import music21

    melody = analysis.melody_notes
    if not melody:
        raise ValueError("No melody notes found in analysis")

    tpb = analysis.ticks_per_beat
    key_str = analysis.key
    tempo = analysis.tempo

    # Parse key
    if key_str.endswith("m"):
        m21_key = music21.key.Key(key_str[:-1].lower(), "minor")
        mode = "minor"
    else:
        m21_key = music21.key.Key(key_str, "major")
        mode = "major"

    # Step 1: Generate chord progression for the melody
    chord_events = _generate_chord_progression(melody, m21_key, tpb)

    # Step 2: Voice the chords in SATB style
    harmony_notes = _voice_chords(chord_events, melody, m21_key, tpb)

    # Step 3: Build output MIDI with string parts
    if output_path is None:
        out_dir = "generated/linear_4096_chord_bpe_hardloss1_PI2/"
        os.makedirs(out_dir, exist_ok=True)
        import time
        timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
        midi_name = "reharmonized"
        output_path = os.path.join(out_dir, f"{midi_name}_{timestamp}.mid")

    result = _write_string_midi(
        melody, harmony_notes, chord_events, tpb, tempo, output_path
    )
    result["chords"] = [f"{ce.numeral}({ce.root})" for ce in chord_events]
    return result


def _generate_chord_progression(
    melody: list[Note],
    m21_key,
    tpb: int,
) -> list[ChordEvent]:
    """Generate a chord progression that fits the melody."""
    import music21

    scale_pitches = [p.midi % 12 for p in m21_key.pitches[:7]]
    tonic_pc = m21_key.tonic.midi % 12
    is_major = m21_key.mode == "major"

    # Select chord map and default chord based on mode
    chord_map = MAJOR_CHORD_MAP if is_major else MINOR_CHORD_MAP
    default_chord = [("I", 1)] if is_major else [("i", 1)]
    progressions = MAJOR_PROGRESSIONS if is_major else MINOR_PROGRESSIONS

    events = []
    prev_numeral = None

    # Harmonic rhythm: major keys use slower harmonic rhythm (one chord per
    # 2 beats minimum) to avoid restless harmony, while minor keys change
    # every beat/note.
    min_chord_dur = 2.0 if is_major else 0.0
    last_chord_beat = -min_chord_dur

    # Group melody notes into harmonic rhythm (one chord per beat or per note)
    for note in melody:
        beat = note.start / tpb
        dur_beats = (note.end - note.start) / tpb

        # In major keys, reuse previous chord if too soon for a change
        if is_major and prev_numeral and (beat - last_chord_beat) < min_chord_dur:
            # Extend previous chord instead of creating a new one
            if events:
                events[-1].duration += dur_beats
                continue

        # Determine scale degree of melody note
        mel_pc = note.pitch % 12
        degree = None
        for i, spc in enumerate(scale_pitches):
            if mel_pc == spc:
                degree = i + 1
                break

        if degree is None:
            # Chromatic note — use dominant or diminished
            degree = 2  # default to V/V7

        # Get candidate chords
        candidates = chord_map.get(degree, default_chord)

        # Score candidates by voice-leading from previous chord
        best_numeral = candidates[0][0]
        best_score = -1

        for numeral, priority in candidates:
            score = priority
            if prev_numeral and numeral in progressions.get(prev_numeral, []):
                score += 3  # bonus for smooth progression
            if numeral == prev_numeral and dur_beats < 2:
                score -= 1  # slight penalty for repeating short chords
            if score > best_score:
                best_score = score
                best_numeral = numeral

        # Realize the chord
        rn = music21.roman.RomanNumeral(best_numeral, m21_key)
        root_name = rn.root().name
        pitch_names = list(rn.pitchNames)

        # Determine bass note (root in bass octave)
        bass_midi = _name_to_bass_midi(root_name, octave=2)

        events.append(ChordEvent(
            numeral=best_numeral,
            root=root_name,
            pitches=pitch_names,
            beat=beat,
            duration=dur_beats,
            bass_note=bass_midi,
        ))
        prev_numeral = best_numeral
        last_chord_beat = beat

    return events


def _voice_chords(
    chord_events: list[ChordEvent],
    melody: list[Note],
    m21_key,
    tpb: int,
) -> list[HarmonyNote]:
    """Voice chords in SATB style with proper voice leading.

    Following Rimsky-Korsakov:
    - Bass: root of chord, rarely more than octave from tenor
    - Tenor: chord tone in mid register
    - Alto: chord tone between melody and tenor
    - Smooth voice leading: minimize motion between voices
    """
    import music21

    notes = []
    prev_alto = 60   # start around middle C
    prev_tenor = 53  # start around F3

    for i, ce in enumerate(chord_events):
        start_tick = int(ce.beat * tpb)
        end_tick = int((ce.beat + ce.duration) * tpb)
        if end_tick <= start_tick:
            continue

        # Get chord pitches
        rn = music21.roman.RomanNumeral(ce.numeral, m21_key)
        chord_pcs = [p.midi % 12 for p in rn.pitches]

        # Bass voice: root, octave 2-3
        bass_midi = ce.bass_note
        notes.append(HarmonyNote(
            pitch=bass_midi, start=start_tick, end=end_tick,
            voice="bass", velocity=75,
        ))

        # Find melody pitch at this time (for spacing)
        mel_pitch = 70  # default
        for mn in melody:
            if mn.start <= start_tick < mn.end:
                mel_pitch = mn.pitch
                break

        # Tenor voice: chord tone in range 48-62, close to previous
        tenor_candidates = []
        for pc in chord_pcs:
            for octave in range(3, 5):
                p = pc + octave * 12
                if 45 <= p <= 62 and p < mel_pitch - 3:
                    tenor_candidates.append(p)

        if tenor_candidates:
            tenor_midi = min(tenor_candidates, key=lambda p: abs(p - prev_tenor))
        else:
            tenor_midi = bass_midi + 12  # octave above bass

        notes.append(HarmonyNote(
            pitch=tenor_midi, start=start_tick, end=end_tick,
            voice="tenor", velocity=70,
        ))

        # Alto voice: chord tone in range 55-72, between melody and tenor
        alto_candidates = []
        for pc in chord_pcs:
            for octave in range(3, 6):
                p = pc + octave * 12
                if tenor_midi < p < mel_pitch and 55 <= p <= 75:
                    alto_candidates.append(p)

        if alto_candidates:
            alto_midi = min(alto_candidates, key=lambda p: abs(p - prev_alto))
        else:
            # If no good candidate, place between tenor and melody
            alto_midi = (tenor_midi + mel_pitch) // 2
            # Snap to nearest chord tone
            best_dist = 999
            for pc in chord_pcs:
                for octave in range(3, 6):
                    p = pc + octave * 12
                    if abs(p - alto_midi) < best_dist and tenor_midi <= p <= mel_pitch:
                        best_dist = abs(p - alto_midi)
                        alto_midi = p

        notes.append(HarmonyNote(
            pitch=alto_midi, start=start_tick, end=end_tick,
            voice="alto", velocity=65,
        ))

        prev_alto = alto_midi
        prev_tenor = tenor_midi

    return notes


def _name_to_bass_midi(name: str, octave: int = 2) -> int:
    """Convert a note name to MIDI pitch at a given octave."""
    name_map = {
        "C": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3,
        "E": 4, "F": 5, "F#": 6, "G-": 6, "G": 7, "G#": 8,
        "A-": 8, "A": 9, "A#": 10, "B-": 10, "B": 11,
    }
    pc = name_map.get(name, 0)
    return pc + (octave + 1) * 12


def _write_string_midi(
    melody: list[Note],
    harmony_notes: list[HarmonyNote],
    chord_events: list[ChordEvent],
    tpb: int,
    tempo: float,
    output_path: str,
) -> dict:
    """Write the re-harmonized score as a string orchestra MIDI.

    Following Rimsky-Korsakov's string rules:
    - Violin I: melody
    - Violin II: alto voice (doubles melody in octave where appropriate)
    - Viola: alto/tenor voice
    - Cello: tenor voice + bass melody
    - Contrabass: bass (octave below cello on sustained notes)
    """
    midi = MidiFile(ticks_per_beat=tpb)

    # Create instruments
    vn1 = Instrument(program=40, name="Violin I")
    vn2 = Instrument(program=40, name="Violin II")
    vla = Instrument(program=41, name="Viola")
    vc = Instrument(program=42, name="Cello")
    cb = Instrument(program=43, name="Contrabass")

    # Violin I: melody
    for n in melody:
        vn1.notes.append(mtkNote(
            velocity=90, pitch=n.pitch, start=n.start, end=n.end,
        ))

    # Violin II: melody doubled an octave below (Rimsky rule: Vn I + Vn II in 8ves)
    for n in melody:
        doubled_pitch = n.pitch - 12
        if 55 <= doubled_pitch <= 90:  # violin range
            vn2.notes.append(mtkNote(
                velocity=75, pitch=doubled_pitch, start=n.start, end=n.end,
            ))

    # Distribute harmony voices
    for hn in harmony_notes:
        if hn.voice == "alto":
            # Alto -> Viola (range 48-93)
            if 48 <= hn.pitch <= 93:
                vla.notes.append(mtkNote(
                    velocity=hn.velocity, pitch=hn.pitch,
                    start=hn.start, end=hn.end,
                ))
        elif hn.voice == "tenor":
            # Tenor -> Cello (range 36-76)
            if 36 <= hn.pitch <= 76:
                vc.notes.append(mtkNote(
                    velocity=hn.velocity, pitch=hn.pitch,
                    start=hn.start, end=hn.end,
                ))
        elif hn.voice == "bass":
            # Bass -> Cello + Contrabass
            # Cello plays the bass note
            cello_pitch = hn.pitch
            if cello_pitch < 36:
                cello_pitch += 12
            if 36 <= cello_pitch <= 76:
                vc.notes.append(mtkNote(
                    velocity=hn.velocity, pitch=cello_pitch,
                    start=hn.start, end=hn.end,
                ))
            # Contrabass doubles (Rimsky: "usually moving in octaves with cellos")
            cb_pitch = hn.pitch
            if cb_pitch < 28:
                cb_pitch += 12
            if 28 <= cb_pitch <= 55:
                cb.notes.append(mtkNote(
                    velocity=hn.velocity - 10, pitch=cb_pitch,
                    start=hn.start, end=hn.end,
                ))

    # Collect all tracks
    track_counts = {}
    for inst in [vn1, vn2, vla, vc, cb]:
        if inst.notes:
            inst.remove_invalid_notes(verbose=False)
            if inst.notes:
                midi.instruments.append(inst)
                track_counts[inst.name] = len(inst.notes)

    # Add tempo
    midi.tempo_changes.append(TempoChange(tempo=tempo, time=0))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    midi.dump(output_path)

    total = sum(track_counts.values())
    return {
        "output_path": output_path,
        "total_notes": total,
        "num_tracks": len(track_counts),
        "tracks": track_counts,
    }
