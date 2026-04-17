"""Piano MIDI analysis: melody/bass extraction, chord detection, key/phrase analysis.

Uses music21 for intelligent voice separation with voice-leading continuity,
replacing the naive skyline algorithm.
"""

import os
import sys
from dataclasses import dataclass, field

from miditoolkit.midi.parser import MidiFile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class Note:
    pitch: int
    start: int   # in ticks (480 ticks/beat)
    end: int
    velocity: int = 90


@dataclass
class Chord:
    root: str        # e.g. "C", "G"
    quality: str     # e.g. "maj", "min", "D7"
    measure: int     # measure index
    label: str = ""  # full label, e.g. "Cmaj", "GD7"


@dataclass
class PianoAnalysis:
    melody_notes: list[Note] = field(default_factory=list)
    bass_notes: list[Note] = field(default_factory=list)
    inner_notes: list[Note] = field(default_factory=list)
    chords: list[Chord] = field(default_factory=list)
    measures: list[tuple[int, int]] = field(default_factory=list)
    key: str = "C"
    tempo: float = 120.0
    time_sig: tuple[int, int] = (4, 4)
    total_measures: int = 0
    phrase_boundaries: list[int] = field(default_factory=list)
    ticks_per_beat: int = 480


# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Melody register threshold: notes below this are unlikely to be melody
MELODY_FLOOR = 60  # middle C — melody should generally be above this


def analyze_piano(midi_path: str) -> PianoAnalysis:
    """Analyze a piano MIDI file and extract musical content.

    Uses music21 for voice separation with voice-leading continuity,
    then converts back to tick-based Note objects for the rest of the pipeline.
    """
    import music21

    midi = MidiFile(midi_path)
    analysis = PianoAnalysis()
    analysis.ticks_per_beat = midi.ticks_per_beat

    # Extract tempo
    if midi.tempo_changes:
        analysis.tempo = midi.tempo_changes[0].tempo

    # Extract time signature
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        analysis.time_sig = (ts.numerator, ts.denominator)

    tpb = midi.ticks_per_beat

    # Collect all notes from all tracks
    all_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            all_notes.append(Note(
                pitch=note.pitch,
                start=note.start,
                end=note.end,
                velocity=note.velocity,
            ))
    all_notes.sort(key=lambda n: (n.start, -n.pitch))

    if not all_notes:
        return analysis

    # Build measure grid
    beats_per_measure = analysis.time_sig[0] * (4 / analysis.time_sig[1])
    ticks_per_measure = int(tpb * beats_per_measure)
    max_tick = max(n.end for n in all_notes)
    num_measures = (max_tick // ticks_per_measure) + 1
    analysis.total_measures = num_measures
    analysis.measures = [
        (i * ticks_per_measure, (i + 1) * ticks_per_measure)
        for i in range(num_measures)
    ]

    # --- Voice separation using music21 ---
    score = music21.converter.parse(midi_path)
    if not score.parts:
        return analysis

    # Chordify all parts together (handles multi-track piano: L/R hands)
    chordified = score.chordify()
    slices = list(chordified.flatten().getElementsByClass(['Chord', 'Note']))

    # Compute the melody register floor dynamically.
    # Strategy: find the largest pitch gap in the upper half of the range.
    # Notes above this gap are melody; notes below are accompaniment.
    all_pitches = []
    for sl in slices:
        if hasattr(sl, 'pitches'):
            all_pitches.extend(p.midi for p in sl.pitches)
        else:
            all_pitches.append(sl.pitch.midi)

    if all_pitches:
        unique_pitches = sorted(set(all_pitches))
        if len(unique_pitches) >= 3:
            # Look for the largest gap in the upper half
            mid_idx = len(unique_pitches) // 2
            upper_pitches = unique_pitches[mid_idx:]
            best_gap = 0
            best_floor = unique_pitches[mid_idx]
            for i in range(len(upper_pitches) - 1):
                gap = upper_pitches[i + 1] - upper_pitches[i]
                if gap > best_gap:
                    best_gap = gap
                    best_floor = upper_pitches[i + 1]
            # If the gap is significant (>= 4 semitones), use it as the floor
            if best_gap >= 4:
                melody_floor = best_floor
            else:
                # No clear gap — use velocity-weighted approach
                # Higher velocity notes are more likely melody
                melody_floor = unique_pitches[len(unique_pitches) * 60 // 100]
        else:
            melody_floor = MELODY_FLOOR
    else:
        melody_floor = MELODY_FLOOR

    # Voice-leading aware soprano/bass extraction
    # Each entry: (offset_beats, midi_pitch, duration_beats, velocity)
    soprano_raw = []
    bass_raw = []
    inner_raw = []

    prev_sop_midi = None

    for sl in slices:
        offset = float(sl.offset)
        dur = float(sl.quarterLength)
        if dur <= 0:
            continue

        if hasattr(sl, 'pitches'):
            pitches = sorted(sl.pitches, key=lambda p: p.midi)
            # Build pitch->velocity map from individual notes in the chord
            pitch_vel = {}
            for i, p in enumerate(pitches):
                try:
                    n = sl[i]
                    vel = n.volume.velocity if n.volume.velocity is not None else 90
                except Exception:
                    vel = 90
                pitch_vel[p.midi] = int(vel)
        else:
            pitches = [sl.pitch]
            vel = sl.volume.velocity if sl.volume.velocity is not None else 90
            pitch_vel = {sl.pitch.midi: int(vel)}

        if not pitches:
            continue

        midi_vals = [p.midi for p in pitches]

        # Bass: always the lowest pitch
        bass_midi = midi_vals[0]
        bass_raw.append((offset, bass_midi, dur, pitch_vel.get(bass_midi, 90)))

        # Soprano: highest pitch with voice-leading continuity
        upper_pitches = [m for m in midi_vals if m >= melody_floor]

        if not upper_pitches:
            # No pitches above melody floor — melody rest
            # All notes are inner voices
            for m in midi_vals[1:]:  # skip bass (already assigned)
                inner_raw.append((offset, m, dur, pitch_vel.get(m, 70)))
            prev_sop_midi = None
            continue

        # Melody is the highest note above the floor.
        highest = max(upper_pitches)
        sop_midi = highest

        soprano_raw.append((offset, sop_midi, dur, pitch_vel.get(sop_midi, 90)))
        prev_sop_midi = sop_midi

        # Inner voices: everything that isn't soprano or bass
        for m in midi_vals:
            if m != sop_midi and m != bass_midi:
                inner_raw.append((offset, m, dur, pitch_vel.get(m, 70)))

    # Convert beats -> ticks and merge consecutive same-pitch segments
    analysis.melody_notes = _beats_to_notes(soprano_raw, tpb)
    analysis.bass_notes = _beats_to_notes(bass_raw, tpb)

    # For inner notes, don't merge — preserve individual notes
    for offset, midi_p, dur, vel in inner_raw:
        start_tick = int(offset * tpb)
        end_tick = int((offset + dur) * tpb)
        if end_tick > start_tick:
            analysis.inner_notes.append(Note(
                pitch=midi_p, start=start_tick, end=end_tick, velocity=vel
            ))

    # Chord analysis per measure
    analysis.chords = _extract_chords_m21(score)

    # Key detection via music21
    analysis.key = _detect_key_m21(score)

    # Phrase detection
    analysis.phrase_boundaries = _detect_phrases(
        analysis.melody_notes, analysis.measures, tpb
    )

    return analysis


def _beats_to_notes(
    raw: list[tuple[float, int, float, int]],
    tpb: int,
) -> list[Note]:
    """Convert (offset_beats, midi_pitch, dur_beats, velocity) to merged Note objects.

    Consecutive segments with the same pitch are merged into single notes.
    Velocity is taken from the first segment in a merged group.
    """
    if not raw:
        return []

    notes = []
    cur_pitch = raw[0][1]
    cur_start = int(raw[0][0] * tpb)
    cur_end = int((raw[0][0] + raw[0][2]) * tpb)
    cur_vel = raw[0][3]

    for offset, midi_p, dur, vel in raw[1:]:
        start_tick = int(offset * tpb)
        end_tick = int((offset + dur) * tpb)

        # Merge if same pitch and continuous (or very small gap)
        if midi_p == cur_pitch and abs(start_tick - cur_end) <= tpb // 8:
            cur_end = max(cur_end, end_tick)
        else:
            if cur_end > cur_start:
                notes.append(Note(pitch=cur_pitch, start=cur_start, end=cur_end, velocity=cur_vel))
            cur_pitch = midi_p
            cur_start = start_tick
            cur_end = end_tick
            cur_vel = vel

    if cur_end > cur_start:
        notes.append(Note(pitch=cur_pitch, start=cur_start, end=cur_end, velocity=cur_vel))

    return notes


def _extract_chords_m21(score) -> list[Chord]:
    """Extract chords per measure using music21's analysis."""

    chords = []
    if not score.parts:
        return chords

    # Chordify all parts together for multi-track analysis
    chordified = score.chordify()

    # Get measure count from first part (all parts share the same structure)
    measures = list(score.parts[0].getElementsByClass('Measure'))
    if not measures:
        return chords

    for mea_idx, measure in enumerate(measures):
        try:
            chord_slices = list(
                chordified.measure(measure.number).flatten().getElementsByClass(['Chord', 'Note'])
            )
        except Exception:
            chords.append(Chord(root="N", quality="A", measure=mea_idx, label="NA"))
            continue

        if not chord_slices:
            chords.append(Chord(root="N", quality="A", measure=mea_idx, label="NA"))
            continue

        # Take the first significant chord in the measure
        for sl in chord_slices:
            if hasattr(sl, 'pitches') and len(sl.pitches) >= 2:
                try:
                    root = sl.root()
                    quality = sl.quality
                    root_name = root.name if root else "N"
                    qual = quality if quality else "maj"
                    chords.append(Chord(
                        root=root_name, quality=qual,
                        measure=mea_idx, label=f"H{root_name}{qual}",
                    ))
                except Exception:
                    chords.append(Chord(root="N", quality="A", measure=mea_idx, label="NA"))
                break
        else:
            chords.append(Chord(root="N", quality="A", measure=mea_idx, label="NA"))

    return chords


def _detect_key_m21(score) -> str:
    """Detect key using music21's built-in key analysis."""
    try:
        key = score.analyze('key')
        name = key.tonic.name
        mode = "m" if key.mode == "minor" else ""
        return f"{name}{mode}"
    except Exception:
        return _detect_key_fallback(score)


def _detect_key_fallback(score) -> str:
    """Fallback key detection using Krumhansl-Schmuckler on all pitches."""

    histogram = [0.0] * 12
    for n in score.flatten().notes:
        if hasattr(n, 'pitches'):
            for p in n.pitches:
                histogram[p.midi % 12] += float(n.quarterLength)
        else:
            histogram[n.pitch.midi % 12] += float(n.quarterLength)

    total = sum(histogram)
    if total == 0:
        return "C"
    histogram = [h / total for h in histogram]

    best_key = "C"
    best_corr = -2.0

    for shift in range(12):
        rotated = histogram[shift:] + histogram[:shift]
        corr_maj = _pearson(rotated, MAJOR_PROFILE)
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = PITCH_NAMES[shift]
        corr_min = _pearson(rotated, MINOR_PROFILE)
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = PITCH_NAMES[shift] + "m"

    return best_key


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


def _detect_phrases(
    melody_notes: list[Note],
    measures: list[tuple[int, int]],
    tpb: int,
) -> list[int]:
    """Detect phrase boundaries using multiple heuristics.

    Criteria (any one triggers a boundary):
    1. Rest: gap >= half a beat between melody notes
    2. Leap: interval >= 7 semitones (perfect 5th)
    3. Dynamic drop: velocity drops by 30%+ from one note to the next
    4. Rhythmic thinning: note density drops significantly at a measure boundary
    """
    if not melody_notes or not measures:
        return [0]

    ticks_per_measure = measures[0][1] - measures[0][0] if measures else tpb * 4
    rest_threshold = tpb // 2  # eighth note rest
    leap_threshold = 7         # perfect 5th
    boundaries = set([0])

    # 1 & 2: Rest and leap detection
    for i in range(1, len(melody_notes)):
        gap = melody_notes[i].start - melody_notes[i - 1].end
        leap = abs(melody_notes[i].pitch - melody_notes[i - 1].pitch)

        if gap >= rest_threshold or leap >= leap_threshold:
            mea_idx = melody_notes[i].start // ticks_per_measure
            if mea_idx < len(measures):
                boundaries.add(mea_idx)

    # 3: Dynamic drop detection
    for i in range(1, len(melody_notes)):
        prev_vel = melody_notes[i - 1].velocity
        curr_vel = melody_notes[i].velocity
        if prev_vel > 0 and curr_vel < prev_vel * 0.7:
            mea_idx = melody_notes[i].start // ticks_per_measure
            if mea_idx < len(measures):
                boundaries.add(mea_idx)

    # 4: Rhythmic density change (notes per measure)
    note_counts = [0] * len(measures)
    for n in melody_notes:
        mea_idx = n.start // ticks_per_measure
        if 0 <= mea_idx < len(note_counts):
            note_counts[mea_idx] += 1

    for i in range(1, len(note_counts)):
        if note_counts[i - 1] > 0 and note_counts[i] > 0:
            ratio = note_counts[i] / note_counts[i - 1]
            if ratio < 0.5 or ratio > 2.0:
                boundaries.add(i)

    # Regular phrasing fallback: every 4 measures for short pieces, 8 for long
    if len(boundaries) < 2 and len(measures) > 4:
        interval = 4 if len(measures) <= 16 else 8
        for i in range(0, len(measures), interval):
            boundaries.add(i)

    result = sorted(boundaries)
    return result
