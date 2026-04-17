#!/usr/bin/env python3
"""Generate training data for Decimus LLM fine-tuning.

Produces Alpaca-format JSONL from the existing knowledge base:
  - Source A: Text Q&A from Rimsky-Korsakov chapters
  - Source B: MusicXML analysis pairs (instrument/doubling analysis)
  - Source C: Good/bad pedagogical MIDI pairs
  - Source D: Structured rules (instrument ranges, doublings)
  - Source E: Orchestration planning pairs (simulated planner output)

Usage:
    python scripts/generate_training_data.py [--output data/training/orchestration_instruct.jsonl]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEXT_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "text"
XML_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "xml" / "examples"
CHAPTER_EXAMPLES_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "xml" / "chapter_examples"
RULES_FILE = PROJECT_ROOT / "data" / "knowledge_base" / "rules" / "xml_analysis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "orchestration_instruct.jsonl"


def write_pair(f, instruction: str, input_text: str, output_text: str):
    """Write one training pair as JSONL."""
    record = {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output_text.strip(),
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Source A: Text Q&A from chapter files
# ---------------------------------------------------------------------------

# Map section headings to question templates
HEADING_QUESTIONS = {
    "melody": "What are the principles for orchestrating {topic}?",
    "harmony": "How should harmonic accompaniment be handled for {topic}?",
    "stringed": "How are stringed instruments used for {topic}?",
    "wind": "How are wind instruments used for {topic}?",
    "brass": "How are brass instruments used for {topic}?",
    "doubl": "What are the rules for doubling in {topic}?",
    "unison": "When should unison writing be used for {topic}?",
    "piano": "How does the piano factor into {topic}?",
    "forte": "How should forte passages be orchestrated for {topic}?",
    "pianissimo": "How should pianissimo passages be orchestrated for {topic}?",
    "tremolo": "When and how should tremolo be used for {topic}?",
    "pizzicato": "When should pizzicato be used for {topic}?",
    "mute": "When should mutes be used for {topic}?",
    "register": "How does register affect {topic}?",
}


def _clean_text(text: str) -> str:
    """Remove page markers and normalize whitespace."""
    text = re.sub(r'-\d+-', '', text)  # Remove page markers like -37-
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple blank lines
    return text.strip()


def _heading_to_question(heading: str) -> str:
    """Convert a section heading to a natural question."""
    topic = heading.strip("# ").strip(".").strip()
    topic_lower = topic.lower()

    for keyword, template in HEADING_QUESTIONS.items():
        if keyword in topic_lower:
            return template.format(topic=topic)

    return f"What does Rimsky-Korsakov teach about {topic}?"


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) pairs based on ## markers."""
    sections = []
    lines = text.split('\n')
    current_heading = ""
    current_body = []

    for line in lines:
        if line.startswith('## '):
            if current_heading and current_body:
                body = '\n'.join(current_body).strip()
                if len(body) > 50:  # Skip tiny sections
                    sections.append((current_heading, body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    # Last section
    if current_heading and current_body:
        body = '\n'.join(current_body).strip()
        if len(body) > 50:
            sections.append((current_heading, body))

    return sections


def _extract_examples(text: str) -> list[tuple[str, str]]:
    """Extract musical example references and their surrounding commentary."""
    pairs = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        # Match example references like "The Tsar's Bride 84.[C]" or "Sheherazade"
        if re.search(r'(?:No\.\s*\d+|[A-Z][a-z]+.*?\d+[\.\[\]])', line):
            # Gather context: this line + next few lines
            context_lines = [line]
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() == '' or lines[j].startswith('##'):
                    break
                context_lines.append(lines[j])

            context = ' '.join(l.strip() for l in context_lines)
            context = _clean_text(context)

            if len(context) > 40:
                pairs.append((
                    f"Describe this orchestration example: {line.strip()[:80]}",
                    context,
                ))

    return pairs


def generate_source_a(f) -> int:
    """Generate text Q&A pairs from chapter files."""
    count = 0

    # Skip complete.txt (it's a superset of the chapter files)
    chapter_files = sorted(TEXT_DIR.glob("ch*.txt")) + [TEXT_DIR / "vol2_examples.txt"]

    for filepath in chapter_files:
        if not filepath.exists():
            continue

        text = filepath.read_text(encoding='utf-8', errors='replace')
        text = _clean_text(text)
        chapter_name = filepath.stem.replace('_', ' ').title()

        # Section-level Q&A
        sections = _split_into_sections(text)
        for heading, body in sections:
            question = _heading_to_question(heading)

            # Trim body to ~800 words max for training
            words = body.split()
            if len(words) > 800:
                body = ' '.join(words[:800]) + '...'

            write_pair(f, question, f"From {chapter_name}:", body)
            count += 1

        # Example-level pairs
        examples = _extract_examples(text)
        for instruction, output_text in examples:
            write_pair(f, instruction, f"From {chapter_name}:", output_text)
            count += 1

    # Generate cross-topic pairs from the preamble/intro sections
    intro_questions = [
        ("What is the purpose of orchestration?",
         "Explain the goals and principles of orchestration according to Rimsky-Korsakov."),
        ("How should melody stand out from accompaniment?",
         "Explain techniques for making melody prominent in orchestration."),
        ("What is the difference between natural and artificial means of orchestration?",
         "Explain the distinction between natural and artificial orchestral techniques."),
    ]

    # Read the first section of ch2 for melody-standing-out context
    ch2 = TEXT_DIR / "ch2_melody.txt"
    if ch2.exists():
        ch2_text = _clean_text(ch2.read_text(encoding='utf-8', errors='replace'))
        # Use the opening paragraphs (before first ##)
        intro = ch2_text.split('## ')[0].strip()
        if intro and len(intro) > 100:
            for question, context in intro_questions:
                write_pair(f, question, context, intro[:1500])
                count += 1

    return count


# ---------------------------------------------------------------------------
# Source B: MusicXML analysis pairs
# ---------------------------------------------------------------------------

def generate_source_b(f) -> int:
    """Generate pairs from MusicXML file analysis."""
    count = 0

    try:
        import music21
    except ImportError:
        print("  [SKIP] Source B: music21 not available")
        return 0

    xml_files = sorted(XML_DIR.glob("*.xml"))
    if not xml_files:
        print("  [SKIP] Source B: no XML files found")
        return 0

    print(f"  Parsing {len(xml_files)} MusicXML files...")
    for i, xml_file in enumerate(xml_files):
        if i % 50 == 0:
            print(f"    {i}/{len(xml_files)}...")
        try:
            score = music21.converter.parse(str(xml_file), forceSource=True)
            parts = score.parts

            if not parts:
                continue

            # Extract instrument info
            instruments_info = []
            for part in parts:
                name = part.partName or "Unknown"
                notes = list(part.flatten().notes)
                if not notes:
                    continue
                pitches = [n.pitch.midi for n in notes if hasattr(n, 'pitch')]
                if not pitches:
                    continue
                instruments_info.append({
                    "name": name,
                    "note_count": len(pitches),
                    "low": min(pitches),
                    "high": max(pitches),
                    "avg": sum(pitches) / len(pitches),
                })

            if len(instruments_info) < 2:
                continue

            # Determine key if possible
            try:
                key = score.analyze('key')
                key_str = f"{key.tonic.name} {key.mode}"
            except Exception:
                key_str = "unknown"

            # Build instrument summary
            inst_summary = "; ".join(
                f"{info['name']} ({info['note_count']} notes, range {info['low']}-{info['high']})"
                for info in instruments_info
            )

            # Pair 1: "What instruments are used in this example?"
            write_pair(
                f,
                "What instruments are used in this orchestral example and what are their roles?",
                f"Example {xml_file.stem} in {key_str}. Instruments: {inst_summary}",
                _describe_instrumentation(instruments_info, key_str),
            )
            count += 1

            # Pair 2: Doubling detection
            doublings = _detect_xml_doublings(parts)
            if doublings:
                doubling_text = "; ".join(
                    f"{d['inst1']} and {d['inst2']} are doubled"
                    for d in doublings[:5]
                )
                write_pair(
                    f,
                    "Which instruments are doubled in this orchestral example?",
                    f"Example {xml_file.stem} in {key_str}.",
                    f"The following doublings are present: {doubling_text}.",
                )
                count += 1

        except Exception:
            continue  # Skip unparseable files

    return count


def _describe_instrumentation(instruments: list[dict], key: str) -> str:
    """Generate a natural-language description of instrumentation."""
    if not instruments:
        return "No instruments found."

    # Sort by average pitch (high to low)
    sorted_inst = sorted(instruments, key=lambda x: -x['avg'])

    lines = [f"This example in {key} uses {len(instruments)} instruments:"]
    for info in sorted_inst:
        register = "high" if info['avg'] > 72 else "low" if info['avg'] < 55 else "mid"
        lines.append(
            f"- {info['name']}: {info['note_count']} notes in the {register} register "
            f"(MIDI {info['low']}-{info['high']})"
        )

    # Identify likely melody (highest avg pitch, significant notes)
    melody_candidate = sorted_inst[0]
    if melody_candidate['note_count'] >= 3:
        lines.append(
            f"\n{melody_candidate['name']} likely carries the melody in the upper register."
        )

    return "\n".join(lines)


def _detect_xml_doublings(parts) -> list[dict]:
    """Detect instrument doublings by comparing onset times."""
    import music21

    part_onsets = {}
    for part in parts:
        name = part.partName or "Unknown"
        onsets = set()
        for n in part.flatten().notes:
            onsets.add(round(float(n.offset), 2))
        if onsets:
            part_onsets[name] = onsets

    doublings = []
    names = list(part_onsets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = len(part_onsets[names[i]] & part_onsets[names[j]])
            total = min(len(part_onsets[names[i]]), len(part_onsets[names[j]]))
            if total > 0 and shared / total > 0.7:
                doublings.append({
                    "inst1": names[i],
                    "inst2": names[j],
                    "shared": shared,
                })

    return sorted(doublings, key=lambda x: -x['shared'])


# ---------------------------------------------------------------------------
# Source C: Good/bad pedagogical MIDI pairs
# ---------------------------------------------------------------------------

def generate_source_c(f) -> int:
    """Generate pairs from good/bad MIDI examples."""
    count = 0

    try:
        from miditoolkit import MidiFile
    except ImportError:
        print("  [SKIP] Source C: miditoolkit not available")
        return 0

    # Good/bad pairs
    pairs = [
        ("bass_inversion_good.mid", "bass_inversion_bad.mid",
         "bass inversion", "proper bass inversions", "incorrect bass inversions that weaken the harmonic foundation"),
        ("seventh_good.mid", "seventh_bad.mid",
         "seventh chord resolution", "proper seventh chord resolution with correct voice leading",
         "incorrect seventh chord resolution with parallel fifths or unresolved tendency tones"),
        ("wide_part_writing.mid", "wide_part_writing_bad.mid",
         "wide part writing", "well-spaced wide voicing with clear register separation",
         "poorly spaced voicing with gaps that create weak sonority"),
    ]

    for good_file, bad_file, topic, good_desc, bad_desc in pairs:
        good_path = CHAPTER_EXAMPLES_DIR / good_file
        bad_path = CHAPTER_EXAMPLES_DIR / bad_file

        if not good_path.exists() or not bad_path.exists():
            continue

        try:
            good_midi = MidiFile(str(good_path))
            bad_midi = MidiFile(str(bad_path))

            good_notes = _midi_to_text(good_midi)
            bad_notes = _midi_to_text(bad_midi)

            # Good example
            write_pair(
                f,
                f"Evaluate this orchestral voicing for {topic}.",
                f"Voicing:\n{good_notes}",
                f"This is a correct example of {good_desc}. The voice leading follows "
                f"proper orchestration principles as taught by Rimsky-Korsakov.",
            )
            count += 1

            # Bad example
            write_pair(
                f,
                f"Evaluate this orchestral voicing for {topic}.",
                f"Voicing:\n{bad_notes}",
                f"This is an incorrect example showing {bad_desc}. "
                f"The correct approach would use {good_desc}.",
            )
            count += 1

        except Exception:
            continue

    # Single examples (textures, part writing, etc.)
    texture_files = [
        ("textures01.mid", "sparse transparent texture with clear voice separation"),
        ("textures02.mid", "moderate density texture with balanced inner voices"),
        ("textures03.mid", "dense orchestral texture with full harmonic support"),
        ("close_part_writing.mid", "close position part writing with voices within an octave"),
        ("consecutive_fifths.mid", "an example demonstrating consecutive fifths to avoid"),
        ("consecutive_octaves.mid", "an example demonstrating consecutive octaves to avoid"),
        ("partial_duplication.mid", "partial duplication where only some voices are doubled"),
        ("upper_pedal.mid", "an upper pedal tone sustained above moving harmonies"),
    ]

    for filename, description in texture_files:
        filepath = CHAPTER_EXAMPLES_DIR / filename
        if not filepath.exists():
            continue

        try:
            midi = MidiFile(str(filepath))
            notes_text = _midi_to_text(midi)

            write_pair(
                f,
                f"Analyze this orchestral texture example.",
                f"MIDI content:\n{notes_text}",
                f"This example demonstrates {description}. "
                f"Understanding these textures is essential for effective orchestration.",
            )
            count += 1

        except Exception:
            continue

    return count


def _midi_to_text(midi) -> str:
    """Convert a MIDI file to a simple text representation."""
    lines = []
    for i, track in enumerate(midi.instruments):
        if not track.notes:
            continue
        name = track.name or f"Track {i}"
        pitches = [n.pitch for n in track.notes]
        lines.append(
            f"{name}: {len(track.notes)} notes, range {min(pitches)}-{max(pitches)}, "
            f"avg pitch {sum(pitches)/len(pitches):.0f}"
        )
    return "\n".join(lines) if lines else "Empty MIDI"


# ---------------------------------------------------------------------------
# Source D: Structured rules (instrument ranges, doublings)
# ---------------------------------------------------------------------------

MIDI_NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def _midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI number to note name (e.g., 60 -> C4)."""
    octave = (midi_num // 12) - 1
    note = MIDI_NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


def generate_source_d(f) -> int:
    """Generate pairs from structured instrument rules."""
    count = 0

    # From INSTRUMENTS dict
    from decimus.instruments import INSTRUMENTS, ENSEMBLES

    # Individual instrument range pairs
    for name, spec in INSTRUMENTS.items():
        low_name = _midi_to_note_name(spec.low)
        high_name = _midi_to_note_name(spec.high)

        write_pair(
            f,
            f"What is the playable range of the {spec.display_name}?",
            "",
            f"The {spec.display_name}'s playable range is {low_name} (MIDI {spec.low}) "
            f"to {high_name} (MIDI {spec.high}). It belongs to the {spec.family} family "
            f"and uses General MIDI program {spec.program}.",
        )
        count += 1

    # "Which instruments can play this pitch" pairs
    test_pitches = [40, 48, 55, 60, 65, 72, 80, 88, 96]
    for pitch in test_pitches:
        note_name = _midi_to_note_name(pitch)
        can_play = [
            spec.display_name for spec in INSTRUMENTS.values()
            if spec.low <= pitch <= spec.high
        ]
        if can_play:
            write_pair(
                f,
                f"Which orchestral instruments can play {note_name} (MIDI {pitch})?",
                "",
                f"The following instruments can play {note_name} (MIDI {pitch}): "
                f"{', '.join(can_play)}.",
            )
            count += 1

    # Ensemble composition pairs
    for ens_name, ens_instruments in ENSEMBLES.items():
        specs = [INSTRUMENTS[k] for k in ens_instruments]
        inst_list = ", ".join(s.display_name for s in specs)
        families = {}
        for s in specs:
            families.setdefault(s.family, []).append(s.display_name)
        family_desc = "; ".join(
            f"{fam}: {', '.join(insts)}" for fam, insts in families.items()
        )

        write_pair(
            f,
            f"What instruments are in the {ens_name} ensemble?",
            "",
            f"The {ens_name} ensemble includes {len(specs)} instruments: {inst_list}. "
            f"Grouped by family: {family_desc}.",
        )
        count += 1

    # Doubling patterns from xml_analysis.json
    if RULES_FILE.exists():
        rules = json.loads(RULES_FILE.read_text())
        doublings = rules.get("doublings", [])

        if doublings:
            # General doubling question
            top_doublings = doublings[:10]
            doubling_lines = "\n".join(
                f"- {d['instrument_1']} + {d['instrument_2']}: "
                f"{d['shared_onsets']} shared onsets"
                for d in top_doublings
            )
            write_pair(
                f,
                "What are the most common instrument doublings in orchestral music?",
                "Based on analysis of Rimsky-Korsakov's orchestration examples.",
                f"The most common doublings (by shared note onsets) are:\n{doubling_lines}\n\n"
                f"Clarinet and piccolo clarinet have the most shared onsets, followed by "
                f"bass clarinet with bassoon. In the strings, violas and cellos commonly "
                f"double each other.",
            )
            count += 1

            # Individual doubling pairs
            for d in doublings:
                write_pair(
                    f,
                    f"Do {d['instrument_1']} and {d['instrument_2']} commonly double each other?",
                    "",
                    f"Yes, {d['instrument_1']} and {d['instrument_2']} frequently play "
                    f"together with {d['shared_onsets']} shared note onsets in "
                    f"Rimsky-Korsakov's examples. This is a well-established doubling practice.",
                )
                count += 1

        # Instrument range data from the analysis
        for inst_name, data in rules.get("instruments", {}).items():
            pitch_range = data.get("pitch_range", [])
            avg = data.get("avg_pitch", 0)
            notes = data.get("note_count", 0)

            if len(pitch_range) == 2 and notes > 0:
                register = "high" if avg > 72 else "low" if avg < 55 else "mid"
                write_pair(
                    f,
                    f"How does Rimsky-Korsakov use the {inst_name} in his orchestration examples?",
                    "",
                    f"In Rimsky-Korsakov's examples, the {inst_name} has {notes} notes "
                    f"spanning MIDI {pitch_range[0]}-{pitch_range[1]} "
                    f"(average pitch {avg:.1f}, {register} register). "
                    f"It is active across {data.get('measures_active', 0)} measures.",
                )
                count += 1

    return count


# ---------------------------------------------------------------------------
# Source E: Orchestration planning pairs
# ---------------------------------------------------------------------------

KEYS = ["C", "Cm", "G", "Gm", "D", "Dm", "A", "Am", "F", "Fm",
        "Bb", "Bbm", "Eb", "Ebm", "Ab"]
TEMPOS = [60, 72, 85, 100, 120, 140]
TIME_SIGS = [(4, 4), (3, 4), (6, 8)]
REGISTERS = ["low", "mid", "high"]
DENSITIES = ["sparse", "moderate", "dense"]
STYLES = ["romantic", "classical", "modern", "film"]
ENSEMBLE_NAMES = ["full", "strings", "chamber", "winds"]

# Average pitches for register simulation
REGISTER_PITCHES = {"low": 48, "mid": 65, "high": 82}
DENSITY_COUNTS = {"sparse": 12, "moderate": 50, "dense": 150}


def _build_planning_output(style_name: str, ensemble_name: str,
                           melody_register: str, bass_register: str,
                           density: str, key: str) -> dict:
    """Build a planning output dict using the actual planner logic."""
    from decimus.styles import get_style
    from decimus.instruments import INSTRUMENTS, ENSEMBLES, get_ensemble

    style = get_style(style_name)
    ensemble = get_ensemble(ensemble_name)
    ensemble_names = {s.name for s in ensemble}

    # Pick melody instrument based on register
    melody_primary = None
    for inst_name in style.melody_instruments:
        if inst_name in ensemble_names:
            melody_primary = inst_name
            break

    # Pick melody doublings
    melody_doublings = [
        name for name in style.melody_doublings
        if name in ensemble_names and name != melody_primary
    ]

    # Pick bass
    bass_primary = None
    for inst_name in style.bass_instruments:
        if inst_name in ensemble_names:
            bass_primary = inst_name
            break

    bass_doublings = [
        name for name in style.bass_doublings
        if name in ensemble_names and name != bass_primary
    ]

    # Pick harmony (exclude already-assigned)
    assigned = {melody_primary} | set(melody_doublings) | {bass_primary} | set(bass_doublings)
    harmony = [
        name for name in style.harmony_instruments
        if name in ensemble_names and name not in assigned
    ]

    # Countermelody
    counter = None
    for name in style.countermelody_instruments:
        if name in ensemble_names and name not in assigned:
            counter = name
            break

    # Build advice text
    mel_inst = INSTRUMENTS[melody_primary] if melody_primary else None
    advice = _generate_advice(style_name, key, melody_register, density,
                              melody_primary, melody_doublings, bass_primary)

    return {
        "melody": {
            "primary": melody_primary,
            "doublings": melody_doublings,
        },
        "bass": {
            "primary": bass_primary,
            "doublings": bass_doublings,
        },
        "harmony": harmony,
        "countermelody": counter,
        "advice": advice,
        "velocity_profile": {
            "melody": 1.0,
            "harmony": 0.7,
            "bass": 0.9,
            "countermelody": 0.85,
        },
    }


def _generate_advice(style: str, key: str, register: str, density: str,
                     melody_inst: str, doublings: list, bass_inst: str) -> str:
    """Generate natural-language orchestration advice."""
    style_desc = {
        "romantic": "rich warm sonority with prominent doublings",
        "classical": "clear transparent textures with minimal doubling",
        "modern": "bold contrasting timbres and unusual combinations",
        "film": "maximalist dramatic power with heavy doublings",
    }

    register_desc = {
        "high": "bright upper register",
        "mid": "warm middle register",
        "low": "deep lower register",
    }

    density_desc = {
        "sparse": "sparse inner voices suggest sustained chords for harmonic support",
        "moderate": "moderate inner voice density allows balanced filling",
        "dense": "dense inner voices require careful distribution to avoid muddiness",
    }

    parts = [
        f"For {key} in {style} style,",
        f"the {register_desc.get(register, 'mid')} melody suits {melody_inst or 'violin_1'}",
    ]
    if doublings:
        parts.append(f"doubled by {', '.join(doublings)} for {style_desc.get(style, 'richness')}")
    parts.append(f"with {bass_inst or 'cello'} anchoring the bass.")
    parts.append(f"The {density_desc.get(density, 'moderate density')}.")

    return " ".join(parts)


def generate_source_e(f) -> int:
    """Generate orchestration planning pairs."""
    count = 0

    for style in STYLES:
        for ensemble in ENSEMBLE_NAMES:
            for mel_reg in REGISTERS:
                for bass_reg in ["low", "mid"]:
                    for density in DENSITIES:
                        # Sample a subset of keys to keep training set manageable
                        for key in KEYS[::3]:  # Every 3rd key
                            mel_pitch = REGISTER_PITCHES[mel_reg]
                            bass_pitch = REGISTER_PITCHES[bass_reg]
                            inner_count = DENSITY_COUNTS[density]

                            input_text = (
                                f"Key: {key}\n"
                                f"Tempo: 85 BPM\n"
                                f"Time signature: 4/4\n"
                                f"Melody register: {mel_reg} (avg pitch {mel_pitch})\n"
                                f"Bass register: {bass_reg} (avg pitch {bass_pitch})\n"
                                f"Inner voice density: {density} ({inner_count} notes)\n"
                                f"Style: {style}\n"
                                f"Ensemble: {ensemble}\n"
                                f"Phrase count: 4\n"
                                f"Total measures: 16\n"
                                f"Chords: i - iv - V7 - i"
                            )

                            try:
                                plan_output = _build_planning_output(
                                    style, ensemble, mel_reg, bass_reg, density, key
                                )
                                output_text = json.dumps(plan_output, indent=2)

                                write_pair(
                                    f,
                                    "Create an orchestration plan for this piano piece.",
                                    input_text,
                                    output_text,
                                )
                                count += 1
                            except Exception:
                                continue

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Decimus LLM training data")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    parser.add_argument("--skip-xml", action="store_true",
                        help="Skip MusicXML parsing (Source B) — much faster")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating Decimus LLM training data -> {args.output}")
    print()

    total = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        print("Source A: Text Q&A from Rimsky-Korsakov chapters...")
        n = generate_source_a(f)
        print(f"  -> {n} pairs")
        total += n

        if not args.skip_xml:
            print("Source B: MusicXML analysis pairs...")
            n = generate_source_b(f)
            print(f"  -> {n} pairs")
            total += n
        else:
            print("Source B: [SKIPPED]")

        print("Source C: Good/bad pedagogical MIDI pairs...")
        n = generate_source_c(f)
        print(f"  -> {n} pairs")
        total += n

        print("Source D: Structured instrument rules...")
        n = generate_source_d(f)
        print(f"  -> {n} pairs")
        total += n

        print("Source E: Orchestration planning pairs...")
        n = generate_source_e(f)
        print(f"  -> {n} pairs")
        total += n

    print()
    print(f"Total: {total} training pairs written to {args.output}")

    # Report file size
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
