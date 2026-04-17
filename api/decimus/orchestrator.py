"""Direct orchestration: distribute original piano notes across orchestral instruments.

This preserves the original melody, harmony, and bass exactly as written,
just re-voiced for a full orchestra.
"""

import os

from miditoolkit.midi.containers import ControlChange, Instrument, TempoChange, TimeSignature
from miditoolkit.midi.containers import Note as mtkNote
from miditoolkit.midi.parser import MidiFile

from .analyzer import Note, PianoAnalysis
from .instruments import clamp_to_range, in_range
from .planner import InstrumentRole, OrchestrationPlan


def _humanize(notes: list[mtkNote], tpb: int, timing_range: int = 0, velocity_range: int = 0) -> None:
    """Add subtle random variations to timing and velocity for a more human feel.

    timing_range: max ticks of offset (positive or negative). 0 = off.
    velocity_range: max velocity deviation. 0 = off.
    """
    if not notes or (timing_range == 0 and velocity_range == 0):
        return

    import random
    rng = random.Random(42)  # deterministic seed for reproducibility

    for note in notes:
        if timing_range > 0:
            offset = rng.randint(-timing_range, timing_range)
            note.start = max(0, note.start + offset)
            note.end = max(note.start + tpb // 16, note.end + offset)
        if velocity_range > 0:
            jitter = rng.randint(-velocity_range, velocity_range)
            note.velocity = max(1, min(127, note.velocity + jitter))


def _add_expression(inst: Instrument, role: InstrumentRole, tpb: int) -> None:
    """Add MIDI CC expression events to make the instrument sound more alive.

    CC11 (Expression): smooth swell curve per note, not just a flat value.
    CC1 (Modulation): graduated vibrato on sustained notes for strings/winds.
    CC64 (Sustain): for piano/harp on long notes.
    """
    if not inst.notes:
        return

    notes = sorted(inst.notes, key=lambda n: n.start)
    family = role.spec.family

    # CC11: Expression — smooth swell: attack → peak → slight decay
    for note in notes:
        dur = note.end - note.start
        base_expr = max(40, min(127, int(note.velocity * 1.1)))

        if dur > tpb:
            # Long note: swell up then sustain
            attack_time = note.start
            peak_time = note.start + min(dur // 4, tpb // 2)
            sustain_time = note.start + dur * 3 // 4
            release_time = note.end

            inst.control_changes.append(ControlChange(number=11, value=max(30, base_expr - 25), time=attack_time))
            inst.control_changes.append(ControlChange(number=11, value=base_expr, time=peak_time))
            inst.control_changes.append(ControlChange(number=11, value=max(35, base_expr - 10), time=sustain_time))
            inst.control_changes.append(ControlChange(number=11, value=max(20, base_expr - 30), time=release_time))
        else:
            # Short note: single value
            inst.control_changes.append(ControlChange(number=11, value=base_expr, time=note.start))

    # CC1: Modulation (vibrato) — graduated onset for strings and woodwinds
    if family in ("strings", "woodwinds"):
        quarter = tpb
        for note in notes:
            dur = note.end - note.start
            if dur > quarter:
                # 3-stage vibrato: none → gentle → moderate
                t0 = note.start
                t1 = note.start + quarter // 3
                t2 = note.start + dur // 3
                t3 = note.start + dur * 2 // 3
                t_end = note.end
                inst.control_changes.append(ControlChange(number=1, value=0, time=t0))
                inst.control_changes.append(ControlChange(number=1, value=15, time=t1))
                inst.control_changes.append(ControlChange(number=1, value=40, time=t2))
                inst.control_changes.append(ControlChange(number=1, value=55, time=t3))
                inst.control_changes.append(ControlChange(number=1, value=0, time=t_end))

    # CC1 for brass: subtle vibrato only on very long notes
    if family == "brass":
        for note in notes:
            dur = note.end - note.start
            if dur > tpb * 2:  # half note or longer
                mid = note.start + dur // 2
                inst.control_changes.append(ControlChange(number=1, value=0, time=note.start))
                inst.control_changes.append(ControlChange(number=1, value=30, time=mid))
                inst.control_changes.append(ControlChange(number=1, value=0, time=note.end))

    # CC64: Sustain pedal for piano and harp on long notes
    if role.spec.name in ("piano", "harp"):
        for note in notes:
            dur = note.end - note.start
            if dur > tpb:
                inst.control_changes.append(ControlChange(number=64, value=127, time=note.start))
                inst.control_changes.append(ControlChange(number=64, value=0, time=note.end))


def _apply_articulations(notes: list[mtkNote], tpb: int) -> None:
    """Apply articulation shaping to notes based on duration and spacing.

    - Staccato: short notes get shortened to 50% duration
    - Legato: connected notes get slight overlap (5% extension)
    - Accents: notes significantly louder than neighbors get velocity boost
    """
    if len(notes) < 2:
        return

    sorted_notes = sorted(notes, key=lambda n: n.start)
    eighth_note = tpb // 2

    # Compute local average velocity (rolling window of 5)
    for i, note in enumerate(sorted_notes):
        # Accent detection: compare to neighbors
        window = sorted_notes[max(0, i - 2):i + 3]
        avg_vel = sum(n.velocity for n in window) / len(window)
        if note.velocity > avg_vel * 1.3 and note.velocity < 120:
            note.velocity = min(127, note.velocity + 8)

    for i in range(len(sorted_notes) - 1):
        cur = sorted_notes[i]
        nxt = sorted_notes[i + 1]
        dur = cur.end - cur.start
        gap = nxt.start - cur.end

        if dur <= eighth_note and gap > 0:
            # Staccato: short note with gap after — shorten
            cur.end = cur.start + max(tpb // 8, dur // 2)
        elif gap <= tpb // 16 and gap >= 0 and dur > eighth_note:
            # Legato: nearly connected notes — slight overlap
            cur.end = min(cur.end + tpb // 16, nxt.start + tpb // 32)


def _enforce_monophonic(notes: list[mtkNote]) -> list[mtkNote]:
    """For monophonic instruments, keep only the highest note at each time position.

    Also trims overlapping notes so that a note ends when the next one begins.
    """
    if not notes:
        return notes

    # Group by start time
    by_start: dict[int, list[mtkNote]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)

    # Keep only the highest pitch at each start time
    kept = []
    for start in sorted(by_start):
        group = by_start[start]
        best = max(group, key=lambda n: n.pitch)
        kept.append(best)

    # Trim overlaps: each note must end before or when the next starts
    for i in range(len(kept) - 1):
        if kept[i].end > kept[i + 1].start:
            kept[i].end = kept[i + 1].start

    return kept


def _compute_phrase_dynamics(
    analysis: PianoAnalysis,
    tpb: int,
) -> list[tuple[int, int, float]]:
    """Compute dynamic level for each phrase region.

    Returns list of (start_tick, end_tick, dynamic_ratio) where dynamic_ratio
    is 0.0 (very quiet) to 1.0 (loudest).
    """
    boundaries = analysis.phrase_boundaries
    measures = analysis.measures
    if not boundaries or not measures:
        return []

    all_notes = analysis.melody_notes + analysis.bass_notes + analysis.inner_notes
    if not all_notes:
        return []

    # Build phrase regions as tick ranges
    ticks_per_measure = measures[0][1] - measures[0][0] if measures else tpb * 4
    regions = []
    for i, start_mea in enumerate(boundaries):
        end_mea = boundaries[i + 1] if i + 1 < len(boundaries) else analysis.total_measures
        start_tick = start_mea * ticks_per_measure
        end_tick = end_mea * ticks_per_measure
        regions.append((start_tick, end_tick))

    # Compute average velocity per region
    region_vels = []
    for start_tick, end_tick in regions:
        vels = [n.velocity for n in all_notes if start_tick <= n.start < end_tick]
        avg_vel = sum(vels) / len(vels) if vels else 70
        region_vels.append(avg_vel)

    # Normalize to 0.0-1.0 range
    min_vel = min(region_vels) if region_vels else 0
    max_vel = max(region_vels) if region_vels else 127
    vel_range = max_vel - min_vel if max_vel > min_vel else 1

    result = []
    for (start_tick, end_tick), avg_vel in zip(regions, region_vels):
        ratio = (avg_vel - min_vel) / vel_range
        result.append((start_tick, end_tick, ratio))

    return result


def _get_active_harmony_count(
    time_tick: int,
    phrase_dynamics: list[tuple[int, int, float]],
    total_harmony: int,
) -> int:
    """Return how many harmony instruments should be active at this time.

    Quiet phrases: ~40% of harmony instruments.
    Loud phrases: all harmony instruments.
    """
    if not phrase_dynamics:
        return total_harmony

    for start, end, ratio in phrase_dynamics:
        if start <= time_tick < end:
            # Scale from 40% to 100% of available instruments
            min_active = max(2, total_harmony * 2 // 5)
            active = min_active + int((total_harmony - min_active) * ratio)
            return max(2, min(total_harmony, active))

    return total_harmony


def _generate_countermelody(
    melody_notes: list[Note],
    counter_spec,
) -> list[mtkNote]:
    """Generate a countermelody line using contrary motion.

    The countermelody:
    - Moves in contrary motion to the melody (when melody rises, counter falls)
    - Stays consonant (3rds, 6ths below the melody)
    - Fits within the counter instrument's range
    - Uses longer note values for contrast (skips some melody notes)
    """
    if len(melody_notes) < 3:
        return []

    counter_notes = []
    # Start a 3rd below the first melody note
    base_interval = -4  # minor 3rd below

    prev_mel_pitch = melody_notes[0].pitch
    prev_counter = melody_notes[0].pitch + base_interval

    for i, note in enumerate(melody_notes):
        # Skip every other note for longer countermelody rhythm
        if i % 2 == 1 and i < len(melody_notes) - 1:
            continue

        # Contrary motion: invert the melody's direction
        mel_motion = note.pitch - prev_mel_pitch
        counter_motion = -mel_motion if mel_motion != 0 else 0

        # Apply motion but keep within consonant intervals
        counter_pitch = prev_counter + counter_motion

        # Snap to consonant interval from melody (3rd or 6th below)
        interval_from_mel = note.pitch - counter_pitch
        if interval_from_mel < 2:
            counter_pitch = note.pitch - 4  # minor 3rd below
        elif interval_from_mel > 10:
            counter_pitch = note.pitch - 9  # major 6th below

        # Clamp to instrument range
        counter_pitch = clamp_to_range(counter_pitch, counter_spec)
        if not in_range(counter_pitch, counter_spec):
            prev_mel_pitch = note.pitch
            continue

        # Determine end time (extend to next countermelody note)
        if i + 2 < len(melody_notes):
            end = melody_notes[i + 2].start
        else:
            end = note.end

        counter_notes.append(mtkNote(
            velocity=max(1, int(note.velocity * 0.7)),
            pitch=counter_pitch,
            start=note.start,
            end=end,
        ))

        prev_mel_pitch = note.pitch
        prev_counter = counter_pitch

    return counter_notes


def _generate_timpani(
    analysis: PianoAnalysis,
    tpb: int,
) -> list[mtkNote]:
    """Generate timpani part: tonic/dominant on strong beats.

    Traditional timpani usage (Rimsky-Korsakov):
    - Tuned to tonic (I) and dominant (V) of the key
    - Plays on beat 1 of most measures (tonic or dominant matching the harmony)
    - Beat 3 in 4/4 gets a lighter hit on climactic passages
    - Rolls (repeated notes) on phrase boundaries for dramatic effect
    - Does NOT play on every beat — restraint is key
    """
    if not analysis.measures:
        return []

    # Parse key to find tonic and dominant pitches in timpani range (40-55)
    key_str = analysis.key
    NOTE_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    ACCIDENTALS = {"b": -1, "#": 1}

    key_clean = key_str.replace("m", "").replace("-", "b")
    root_pc = NOTE_MAP.get(key_clean[0], 0)
    if len(key_clean) > 1 and key_clean[1] in ACCIDENTALS:
        root_pc = (root_pc + ACCIDENTALS[key_clean[1]]) % 12

    dominant_pc = (root_pc + 7) % 12  # perfect fifth above

    # Find tonic and dominant in timpani range (MIDI 40-55)
    def _find_in_range(pc: int) -> int:
        for octave in range(3, 5):  # C3=48, C4=60
            p = pc + (octave + 1) * 12
            if 40 <= p <= 55:
                return p
        # Fallback: closest in range
        p = pc + 48  # try octave 3
        if p < 40:
            p += 12
        if p > 55:
            p -= 12
        return max(40, min(55, p))

    tonic_pitch = _find_in_range(root_pc)
    dominant_pitch = _find_in_range(dominant_pc)

    notes: list[mtkNote] = []
    beats_per_measure = analysis.time_sig[0]
    phrase_bounds = set(analysis.phrase_boundaries) if analysis.phrase_boundaries else set()

    # Determine which chords are tonic vs dominant (from analysis)
    chord_roots: dict[int, int] = {}  # measure -> root pitch class
    for chord in analysis.chords:
        if hasattr(chord, 'root') and hasattr(chord, 'measure'):
            chord_roots[chord.measure] = chord.root % 12 if isinstance(chord.root, int) else 0

    for i, (m_start, m_end) in enumerate(analysis.measures):
        beat_len = (m_end - m_start) // max(1, beats_per_measure)
        note_dur = beat_len  # quarter note duration

        # Choose pitch: tonic or dominant based on harmony
        chord_root = chord_roots.get(i, root_pc)
        # If chord root is closer to dominant, use dominant pitch
        dist_to_tonic = min(abs(chord_root - root_pc), 12 - abs(chord_root - root_pc))
        dist_to_dom = min(abs(chord_root - dominant_pc), 12 - abs(chord_root - dominant_pc))
        pitch = dominant_pitch if dist_to_dom < dist_to_tonic else tonic_pitch

        # Phrase boundary: roll (two quick hits) for dramatic effect
        if i in phrase_bounds:
            roll_dur = beat_len // 4
            notes.append(mtkNote(pitch=pitch, start=m_start, end=m_start + roll_dur, velocity=90))
            notes.append(mtkNote(pitch=pitch, start=m_start + roll_dur, end=m_start + roll_dur * 2, velocity=85))
            notes.append(mtkNote(pitch=pitch, start=m_start + roll_dur * 2, end=m_start + beat_len, velocity=95))
            continue

        # Beat 1: standard hit (every other measure, or every measure in loud passages)
        # Play every measure for now, let dynamics shape it
        notes.append(mtkNote(pitch=pitch, start=m_start, end=m_start + note_dur, velocity=75))

        # Beat 3 in 4/4: lighter accent only every 4th measure (restraint)
        if beats_per_measure >= 4 and i % 4 == 0:
            t3 = m_start + 2 * beat_len
            notes.append(mtkNote(pitch=pitch, start=t3, end=t3 + beat_len // 2, velocity=55))

    return notes


def _generate_percussion(
    analysis: PianoAnalysis,
    style: str,
    tpb: int,
) -> list[mtkNote]:
    """Generate percussion hits based on beat positions and phrase dynamics.

    GM drum map pitches:
        35 = Acoustic Bass Drum
        38 = Acoustic Snare
        42 = Closed Hi-Hat
        46 = Open Hi-Hat
        49 = Crash Cymbal 1
        51 = Ride Cymbal
        56 = Cowbell
    """
    if not analysis.measures:
        return []

    notes = []
    beats_per_measure = analysis.time_sig[0]

    # Determine pattern based on style
    if style == "rhythmic":
        # Stravinsky/Zimmer: driving rhythm
        for m_start, m_end in analysis.measures:
            beat_len = (m_end - m_start) // beats_per_measure
            for beat in range(beats_per_measure):
                t = m_start + beat * beat_len
                # Kick on 1, 3
                if beat % 2 == 0:
                    notes.append(mtkNote(pitch=35, start=t, end=t + beat_len // 2, velocity=85))
                # Snare on 2, 4
                if beat % 2 == 1:
                    notes.append(mtkNote(pitch=38, start=t, end=t + beat_len // 2, velocity=75))
                # Hi-hat on every beat
                notes.append(mtkNote(pitch=42, start=t, end=t + beat_len // 4, velocity=50))
                # Hi-hat on off-beats too
                t_off = t + beat_len // 2
                if t_off < m_end:
                    notes.append(mtkNote(pitch=42, start=t_off, end=t_off + beat_len // 4, velocity=40))
    elif style == "dramatic":
        # Beethoven/Tchaikovsky/Mahler/Williams: timpani rolls + crashes on climaxes
        phrase_bounds = set(analysis.phrase_boundaries) if analysis.phrase_boundaries else set()
        for i, (m_start, m_end) in enumerate(analysis.measures):
            beat_len = (m_end - m_start) // beats_per_measure
            measure_idx = i
            # Crash cymbal on phrase starts
            if measure_idx in phrase_bounds:
                notes.append(mtkNote(pitch=49, start=m_start, end=m_start + beat_len, velocity=95))
            # Kick on beat 1 of every measure
            notes.append(mtkNote(pitch=35, start=m_start, end=m_start + beat_len // 2, velocity=70))
            # Beat 3 kick (in 4/4)
            if beats_per_measure >= 4:
                t3 = m_start + 2 * beat_len
                notes.append(mtkNote(pitch=35, start=t3, end=t3 + beat_len // 2, velocity=55))
    elif style == "sparse":
        # Ravel/Beethoven: occasional accents
        phrase_bounds = set(analysis.phrase_boundaries) if analysis.phrase_boundaries else set()
        for i, (m_start, m_end) in enumerate(analysis.measures):
            beat_len = (m_end - m_start) // beats_per_measure
            # Only on phrase boundaries or every 4th measure
            if i in phrase_bounds or i % 4 == 0:
                notes.append(mtkNote(pitch=35, start=m_start, end=m_start + beat_len // 2, velocity=60))
            # Crash only on phrase starts
            if i in phrase_bounds:
                notes.append(mtkNote(pitch=49, start=m_start, end=m_start + beat_len, velocity=80))

    return notes


def _scale_velocity(velocity: int, scale: float) -> int:
    """Scale a source velocity by a role's velocity_scale, clamping to MIDI range."""
    return max(1, min(127, int(velocity * scale)))


def orchestrate_direct(
    analysis: PianoAnalysis,
    plan: OrchestrationPlan,
    output_path: str,
    ticks_per_beat: int = 480,
) -> dict:
    """Orchestrate by directly assigning original piano notes to orchestral instruments.

    Returns summary dict with note counts per track.
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    # Write tempo and time signature from analysis
    midi.tempo_changes = [TempoChange(tempo=analysis.tempo, time=0)]
    midi.time_signature_changes = [
        TimeSignature(
            numerator=analysis.time_sig[0],
            denominator=analysis.time_sig[1],
            time=0,
        )
    ]

    # Separate roles by function
    harmony_roles = [r for r in plan.roles if r.role == "harmony"]
    melody_roles = [r for r in plan.roles if r.role in ("melody", "doubling")]
    bass_roles = [r for r in plan.roles if r.role == "bass"]
    counter_roles = [r for r in plan.roles if r.role == "countermelody"]
    timpani_roles = [r for r in plan.roles if r.role == "timpani"]

    # Build Instrument objects per role
    instruments: dict[int, Instrument] = {}
    for role in plan.roles:
        inst = Instrument(
            program=role.spec.program % 128,
            is_drum=role.spec.is_drum,
            name=role.spec.display_name,
        )
        instruments[role.track_id] = inst

    # 1. Melody -> all melody roles (primary + doublings)
    for role in melody_roles:
        inst = instruments[role.track_id]
        for note in analysis.melody_notes:
            pitch = clamp_to_range(note.pitch, role.spec)
            if not in_range(pitch, role.spec):
                continue
            inst.notes.append(mtkNote(
                velocity=_scale_velocity(note.velocity, role.velocity_scale),
                pitch=pitch,
                start=note.start,
                end=note.end,
            ))

    # 2. Bass -> all bass roles (with octave doubling for contrabass etc.)
    for role in bass_roles:
        inst = instruments[role.track_id]
        for note in analysis.bass_notes:
            pitch = clamp_to_range(note.pitch, role.spec)
            if not in_range(pitch, role.spec):
                continue
            inst.notes.append(mtkNote(
                velocity=_scale_velocity(note.velocity, role.velocity_scale),
                pitch=pitch,
                start=note.start,
                end=note.end,
            ))

    # 2b. Countermelody -> generate independent contrary-motion line
    for role in counter_roles:
        inst = instruments[role.track_id]
        counter_notes = _generate_countermelody(analysis.melody_notes, role.spec)
        inst.notes.extend(counter_notes)

    # 2c. Percussion -> generate rhythmic pattern based on style
    perc_roles = [r for r in plan.roles if r.role == "percussion"]
    if perc_roles:
        perc_style = getattr(plan.style, "percussion_style", None) or "sparse"
        perc_notes = _generate_percussion(analysis, perc_style, ticks_per_beat)
        for role in perc_roles:
            inst = instruments[role.track_id]
            for n in perc_notes:
                inst.notes.append(mtkNote(
                    velocity=_scale_velocity(n.velocity, role.velocity_scale),
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end,
                ))

    # 2d. Timpani -> tonic/dominant on strong beats (beat 1 and 3)
    #     Following orchestration tradition: timpani reinforces harmonic rhythm,
    #     not random inner voice notes
    if timpani_roles:
        timpani_notes = _generate_timpani(analysis, ticks_per_beat)
        for role in timpani_roles:
            inst = instruments[role.track_id]
            for n in timpani_notes:
                inst.notes.append(mtkNote(
                    velocity=_scale_velocity(n.velocity, role.velocity_scale),
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end,
                ))

    # 3. Inner voices -> distribute across harmony instruments
    #    Following orchestration principles:
    #    - Strings are the backbone of harmony (prefer for sustained notes)
    #    - Winds add color (use selectively, not constantly)
    #    - Sort by register: high instruments for high notes
    #    - Voice-leading: minimize pitch jumps within each instrument
    #    - Texture variation: fewer instruments in quiet passages
    all_harmony = harmony_roles
    phrase_dynamics = _compute_phrase_dynamics(analysis, ticks_per_beat)

    # Sort harmony by register (low to high) for natural voice assignment
    all_harmony_sorted = sorted(all_harmony, key=lambda r: (r.spec.low + r.spec.high) / 2)
    # Separate strings (backbone) from winds/brass (color)
    string_harmony = [r for r in all_harmony_sorted if r.spec.family == "strings"]
    color_harmony = [r for r in all_harmony_sorted if r.spec.family != "strings"]

    if all_harmony:
        last_pitch: dict[int, int] = {}

        time_slots: dict[int, list[Note]] = {}
        for note in analysis.inner_notes:
            time_slots.setdefault(note.start, []).append(note)

        for slot in sorted(time_slots.keys()):
            notes = sorted(time_slots[slot], key=lambda n: -n.pitch)

            active_count = _get_active_harmony_count(slot, phrase_dynamics, len(all_harmony))
            # Prioritize strings, add winds for louder passages
            if active_count <= len(string_harmony):
                active_harmony = string_harmony[:active_count]
            else:
                active_harmony = string_harmony + color_harmony[:active_count - len(string_harmony)]

            assigned_roles: set[int] = set()

            for note in notes:
                best_role = None
                best_pitch = note.pitch
                best_cost = float('inf')

                for role in active_harmony:
                    if role.track_id in assigned_roles:
                        continue
                    pitch = clamp_to_range(note.pitch, role.spec)
                    if not in_range(pitch, role.spec):
                        continue

                    # Voice-leading cost
                    if role.track_id in last_pitch:
                        cost = abs(pitch - last_pitch[role.track_id])
                    else:
                        center = (role.spec.low + role.spec.high) // 2
                        cost = abs(pitch - center) * 0.5

                    # Prefer strings for sustained harmony (slight bonus)
                    if role.spec.family == "strings":
                        cost -= 2

                    if cost < best_cost:
                        best_cost = cost
                        best_role = role
                        best_pitch = pitch

                if best_role is not None:
                    instruments[best_role.track_id].notes.append(mtkNote(
                        velocity=_scale_velocity(note.velocity, best_role.velocity_scale),
                        pitch=best_pitch,
                        start=note.start,
                        end=note.end,
                    ))
                    last_pitch[best_role.track_id] = best_pitch
                    assigned_roles.add(best_role.track_id)
                else:
                    # Fallback to first available string, then any
                    fallback = string_harmony[0] if string_harmony else all_harmony[0]
                    pitch = clamp_to_range(note.pitch, fallback.spec)
                    instruments[fallback.track_id].notes.append(mtkNote(
                        velocity=_scale_velocity(note.velocity, fallback.velocity_scale),
                        pitch=pitch,
                        start=note.start,
                        end=note.end,
                    ))
                    last_pitch[fallback.track_id] = pitch

    # Enforce monophonic instruments: keep only the highest note at each time
    for role in plan.roles:
        if role.spec.is_monophonic:
            inst = instruments[role.track_id]
            inst.notes = _enforce_monophonic(inst.notes)

    # Apply articulation shaping (staccato, legato, accents)
    for role in plan.roles:
        inst = instruments[role.track_id]
        if inst.notes:
            _apply_articulations(inst.notes, ticks_per_beat)

    # Re-enforce monophonic after articulation (legato may have added overlaps)
    for role in plan.roles:
        if role.spec.is_monophonic:
            inst = instruments[role.track_id]
            inst.notes = _enforce_monophonic(inst.notes)

    # Add MIDI expression controllers (CC11, CC1, CC64)
    for role in plan.roles:
        inst = instruments[role.track_id]
        if inst.notes:
            _add_expression(inst, role, ticks_per_beat)

    # Humanize: subtle timing and velocity variations for realism
    # Melody gets less jitter (must stay tight), inner voices get more
    for role in plan.roles:
        inst = instruments[role.track_id]
        if inst.notes:
            if role.role in ("melody", "doubling"):
                _humanize(inst.notes, ticks_per_beat, timing_range=ticks_per_beat // 48, velocity_range=3)
            elif role.role == "bass":
                _humanize(inst.notes, ticks_per_beat, timing_range=ticks_per_beat // 32, velocity_range=4)
            else:
                _humanize(inst.notes, ticks_per_beat, timing_range=ticks_per_beat // 24, velocity_range=5)

    # Write MIDI
    track_counts = {}
    track_notes: dict[str, list[dict]] = {}
    for role in plan.roles:
        inst = instruments[role.track_id]
        if inst.notes:
            inst.remove_invalid_notes(verbose=False)
            if inst.notes:
                midi.instruments.append(inst)
                name = role.spec.display_name
                track_counts[name] = len(inst.notes)
                track_notes[name] = [
                    {
                        "pitch": n.pitch,
                        "start": n.start,
                        "end": n.end,
                        "velocity": n.velocity,
                    }
                    for n in inst.notes
                ]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    midi.dump(output_path)

    total_notes = sum(track_counts.values())
    return {
        "total_notes": total_notes,
        "tracks": track_counts,
        "track_notes": track_notes,
        "num_tracks": len(track_counts),
        "output_path": output_path,
    }
