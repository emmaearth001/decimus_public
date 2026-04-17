#!/usr/bin/env python3
"""Generate LLM training pairs from real SymphonyNet orchestral MIDI scores.

Replaces the circular Source E with ground-truth orchestration decisions
extracted from 46K real symphonic scores.

For each MIDI file:
  1. Parse all instrument tracks
  2. Detect key, tempo, density
  3. Identify roles: melody, bass, harmony, countermelody, doublings
  4. Output Alpaca-format training pairs

Usage:
    source venv312/bin/activate
    PYTHONPATH=src python scripts/generate_symphonynet_pairs.py
    PYTHONPATH=src python scripts/generate_symphonynet_pairs.py --limit 1000 --workers 4
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from miditoolkit import MidiFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "data" / "symphonynet" / "SymphonyNet_Dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "symphonynet_pairs.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Instrument name normalization ──────────────────────────────────────────
# SymphonyNet has names in English, Spanish, Italian, German, French, etc.

FAMILY_KEYWORDS = {
    "strings": [
        "violin", "violín", "violon", "violine", "vln", "vl.",
        "viola", "vla", "bratsche", "alto",
        "cello", "violoncello", "violoncel", "vlc", "vc.",
        "contrabajo", "contrabass", "kontrabass", "double bass", "bass", "cb.", "kb.",
        "basso", "bassi",
        "harp", "harpe", "arpa", "harfe",
    ],
    "woodwinds": [
        "flute", "flauta", "flauto", "flöte", "fl.",
        "piccolo", "picc", "ottavino", "flauto piccolo", "petite",
        "oboe", "oboi", "hautbois", "ob.",
        "english horn", "cor anglais", "corno inglese",
        "clarinet", "clarinete", "clarinetto", "klarinette", "cl.",
        "bassoon", "fagot", "fagott", "basson", "fg.",
        "contrabassoon", "contrafagot", "kontrafagott",
    ],
    "brass": [
        "horn", "trompa", "corno", "cor ", "cor.", "hn.",
        "trumpet", "trompeta", "tromba", "trompette", "trp", "tr.",
        "trombone", "trombón", "posaune", "tbn",
        "tuba", "tb.",
    ],
    "percussion": [
        "timpani", "timbal", "timbale", "pauke", "kettle",
        "percussion", "percusión",
        "triangle", "triángulo", "triangolo",
        "cymbal", "platillo", "piatti", "becken",
        "bass drum", "gran caja", "gran cassa", "große trommel",
        "snare", "caja", "tamburo", "kleine trommel",
        "glockenspiel", "campan",
        "xylophon", "vibraphon",
        "tambourine", "pandereta", "tamburino",
        "tam-tam", "gong",
    ],
}

# Specific instrument mapping for role detection
MELODY_PROGRAMS = {
    # GM programs commonly carrying melody
    73, 74,  # flute, piccolo
    68, 69,  # oboe, english horn
    71, 72,  # clarinet, bass clarinet
    40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51,  # strings
    56, 57,  # trumpet
    60, 61,  # horn
}

BASS_PROGRAMS = {
    42, 43,  # cello, contrabass
    70,      # bassoon
    57, 58,  # tuba, trombone
}


def classify_family(name: str, program: int, is_drum: bool) -> str:
    """Classify an instrument track into a family."""
    if is_drum:
        return "percussion"
    name_lower = name.lower()
    for family, keywords in FAMILY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return family
    # Fallback by GM program
    if 0 <= program <= 7:
        return "keyboard"
    if 24 <= program <= 31:
        return "strings"  # guitar family
    if 32 <= program <= 39:
        return "strings"  # bass
    if 40 <= program <= 51:
        return "strings"
    if 56 <= program <= 63:
        return "brass"
    if 64 <= program <= 79:
        return "woodwinds"
    if program >= 112:
        return "percussion"
    return "other"


def normalize_instrument_name(name: str) -> str:
    """Simplify instrument name for training output."""
    name_lower = name.lower().strip()
    # Map common variants to English
    mappings = [
        (["violin 1", "violín 1", "vln 1", "vl. 1", "violino 1", "violine 1",
          "violins 1", "violines 1", "violini 1"], "violin_1"),
        (["violin 2", "violín 2", "vln 2", "vl. 2", "violino 2", "violine 2",
          "violins 2", "violines 2", "violini 2"], "violin_2"),
        (["viola", "violas", "bratsche", "alto"], "viola"),
        (["cello", "violoncello", "violoncel", "vlc", "cellos"], "cello"),
        (["contrabajo", "contrabass", "kontrabass", "double bass", "basses",
          "contrebasse", "bassi"], "contrabass"),
        (["flauta", "flauto", "flöte", "flute", "flutes", "flautas"], "flute"),
        (["piccolo", "ottavino", "flauto piccolo", "petite flûte"], "piccolo"),
        (["oboe", "oboi", "hautbois"], "oboe"),
        (["english horn", "cor anglais", "corno inglese"], "english_horn"),
        (["clarinet", "clarinete", "clarinetto", "klarinette"], "clarinet"),
        (["bassoon", "fagot", "fagott", "basson"], "bassoon"),
        (["trompa", "corno", "horn", "cor "], "horn"),
        (["trumpet", "trompeta", "tromba", "trompette"], "trumpet"),
        (["trombone", "trombón", "posaune"], "trombone"),
        (["tuba"], "tuba"),
        (["timpani", "timbal", "timbale", "pauke"], "timpani"),
        (["harp", "harpe", "arpa", "harfe"], "harp"),
    ]
    for variants, canonical in mappings:
        for v in variants:
            if v in name_lower:
                # Handle numbered instruments (e.g., "Flauta 1" → "flute_1")
                for suffix in ["1", "2", "3", "4"]:
                    if suffix in name_lower and canonical not in ("violin_1", "violin_2"):
                        return f"{canonical}_{suffix}"
                return canonical
    return name_lower.replace(" ", "_")[:30]


def detect_key_from_midi(midi: MidiFile) -> str:
    """Estimate key from note distribution using Krumhansl-Schmuckler."""
    # Major and minor profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Count pitch classes weighted by duration
    pitch_class_dur = [0.0] * 12
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            dur = max(1, note.end - note.start)
            pitch_class_dur[note.pitch % 12] += dur

    total = sum(pitch_class_dur)
    if total == 0:
        return "C"

    # Normalize
    dist = [x / total for x in pitch_class_dur]

    # Correlate with each key
    best_key = "C"
    best_corr = -2.0
    for shift in range(12):
        rotated = dist[shift:] + dist[:shift]
        for profile, mode in [(major_profile, ""), (minor_profile, "m")]:
            mean_r = sum(rotated) / 12
            mean_p = sum(profile) / 12
            num = sum((r - mean_r) * (p - mean_p) for r, p in zip(rotated, profile))
            den_r = sum((r - mean_r) ** 2 for r in rotated) ** 0.5
            den_p = sum((p - mean_p) ** 2 for p in profile) ** 0.5
            if den_r * den_p > 0:
                corr = num / (den_r * den_p)
            else:
                corr = 0
            if corr > best_corr:
                best_corr = corr
                best_key = f"{note_names[shift]}{mode}"

    return best_key


def detect_tempo(midi: MidiFile) -> float:
    """Get tempo from MIDI tempo changes."""
    if midi.tempo_changes:
        return midi.tempo_changes[0].tempo
    return 120.0


def analyze_track(inst) -> dict | None:
    """Analyze a single instrument track."""
    notes = inst.notes
    if len(notes) < 3:
        return None

    pitches = [n.pitch for n in notes]
    velocities = [n.velocity for n in notes]
    durations = [n.end - n.start for n in notes]
    onsets = sorted(set(n.start for n in notes))

    return {
        "name": inst.name or "Unknown",
        "program": inst.program,
        "is_drum": inst.is_drum,
        "note_count": len(notes),
        "avg_pitch": sum(pitches) / len(pitches),
        "min_pitch": min(pitches),
        "max_pitch": max(pitches),
        "avg_velocity": sum(velocities) / len(velocities),
        "avg_duration": sum(durations) / len(durations),
        "onsets": onsets,
        "onset_set": set(onsets),
        "total_duration": max(n.end for n in notes) - min(n.start for n in notes),
        "density": len(notes) / max(1, (max(n.end for n in notes) - min(n.start for n in notes))),
    }


def detect_roles(tracks: list[dict]) -> dict:
    """Detect orchestration roles from analyzed tracks."""
    if not tracks:
        return {}

    roles = {}

    # Sort by average pitch descending
    by_pitch = sorted(tracks, key=lambda t: -t["avg_pitch"])
    # Sort by note count descending (activity)
    by_activity = sorted(tracks, key=lambda t: -t["note_count"])

    # ── Melody: highest-pitched active instrument (non-percussion) ──
    melody = None
    for t in by_pitch:
        family = classify_family(t["name"], t["program"], t["is_drum"])
        if family == "percussion":
            continue
        # Must have reasonable activity (top 50% by note count)
        median_count = sorted(tt["note_count"] for tt in tracks)[len(tracks) // 2]
        if t["note_count"] >= median_count * 0.5:
            melody = t
            break

    if melody:
        roles[melody["name"]] = "melody"

    # ── Bass: lowest-pitched active instrument ──
    bass = None
    for t in reversed(by_pitch):
        family = classify_family(t["name"], t["program"], t["is_drum"])
        if family == "percussion":
            continue
        if t.get("name") == (melody or {}).get("name"):
            continue
        if t["avg_pitch"] < 60 and t["note_count"] >= 5:
            bass = t
            break

    if bass:
        roles[bass["name"]] = "bass"

    # ── Doublings: instruments with >70% onset overlap ──
    assigned = {melody and melody["name"], bass and bass["name"]} - {None}

    for i, t1 in enumerate(tracks):
        if t1["name"] in assigned:
            continue
        for t2 in tracks[i + 1:]:
            if t2["name"] in assigned:
                continue
            shared = len(t1["onset_set"] & t2["onset_set"])
            total = min(len(t1["onset_set"]), len(t2["onset_set"]))
            if total > 10 and shared / total > 0.7:
                # The one closer in pitch to melody is doubling
                if melody:
                    d1 = abs(t1["avg_pitch"] - melody["avg_pitch"])
                    d2 = abs(t2["avg_pitch"] - melody["avg_pitch"])
                    doubler = t1 if d1 < d2 else t2
                    other = t2 if d1 < d2 else t1
                    if doubler["name"] not in roles:
                        roles[doubler["name"]] = "doubling"
                    if other["name"] not in roles:
                        roles[other["name"]] = "harmony"
                break

    # ── Countermelody: second most active non-bass, non-melody ──
    for t in by_activity:
        if t["name"] in roles:
            continue
        family = classify_family(t["name"], t["program"], t["is_drum"])
        if family == "percussion":
            roles[t["name"]] = "percussion"
            continue
        if melody and abs(t["avg_pitch"] - melody["avg_pitch"]) > 5:
            roles[t["name"]] = "countermelody"
            break

    # ── Everything else is harmony ──
    for t in tracks:
        if t["name"] not in roles:
            family = classify_family(t["name"], t["program"], t["is_drum"])
            if family == "percussion":
                roles[t["name"]] = "percussion"
            else:
                roles[t["name"]] = "harmony"

    return roles


def process_midi(midi_path: str) -> dict | None:
    """Process a single MIDI file into a training pair."""
    try:
        midi = MidiFile(midi_path)
    except Exception:
        return None

    # Analyze tracks
    tracks = []
    for inst in midi.instruments:
        info = analyze_track(inst)
        if info:
            tracks.append(info)

    # Need at least 5 instrument tracks for meaningful orchestration
    # (filters out choral, piano, chamber duets)
    if len(tracks) < 5:
        return None

    # Must have at least 2 different families (not just strings or just voices)
    track_families = set()
    for t in tracks:
        fam = classify_family(t["name"], t["program"], t["is_drum"])
        track_families.add(fam)
    orchestral_families = track_families & {"strings", "woodwinds", "brass", "percussion"}
    if len(orchestral_families) < 2:
        return None

    # Detect musical properties
    key = detect_key_from_midi(midi)
    tempo = detect_tempo(midi)
    total_notes = sum(t["note_count"] for t in tracks)
    non_perc = [t for t in tracks if not t["is_drum"]]
    if not non_perc:
        return None

    mel_pitches = [t["avg_pitch"] for t in non_perc]
    mel_avg = max(mel_pitches)
    bass_avg = min(mel_pitches)
    inner_count = total_notes - max(t["note_count"] for t in non_perc) - min(t["note_count"] for t in non_perc)

    mel_register = "low" if mel_avg < 55 else ("high" if mel_avg > 75 else "mid")
    bass_register = "low" if bass_avg < 55 else ("high" if bass_avg > 75 else "mid")
    density = "sparse" if inner_count < 50 else ("dense" if inner_count > 500 else "moderate")

    # Detect roles
    roles = detect_roles(tracks)
    if not roles:
        return None

    # Count families
    families = defaultdict(list)
    for t in tracks:
        family = classify_family(t["name"], t["program"], t["is_drum"])
        families[family].append(normalize_instrument_name(t["name"]))

    # Determine ensemble type
    family_set = set(families.keys()) - {"other", "keyboard"}
    if family_set >= {"strings", "woodwinds", "brass"}:
        ensemble = "full"
    elif family_set == {"strings"}:
        ensemble = "strings"
    elif len(tracks) <= 6:
        ensemble = "chamber"
    elif "woodwinds" in family_set and "strings" not in family_set:
        ensemble = "winds"
    else:
        ensemble = "full"

    # Estimate total measures
    tpb = midi.ticks_per_beat or 480
    max_tick = max(n.end for inst in midi.instruments for n in inst.notes) if midi.instruments else 0
    total_measures = max(1, int(max_tick / (tpb * 4)))

    # ── Build input text (same format as Source E) ──
    input_text = (
        f"Create an orchestration plan for this piano piece.\n\n"
        f"Key: {key}\n"
        f"Tempo: {tempo:.0f} BPM\n"
        f"Time signature: 4/4\n"
        f"Melody register: {mel_register} (avg pitch {mel_avg:.0f})\n"
        f"Bass register: {bass_register} (avg pitch {bass_avg:.0f})\n"
        f"Inner voice density: {density} ({inner_count} notes)\n"
        f"Style: orchestral\n"
        f"Ensemble: {ensemble}\n"
        f"Phrase count: {max(1, total_measures // 8)}\n"
        f"Total measures: {total_measures}\n"
    )

    # ── Build output (the real orchestration decision) ──
    melody_inst = None
    melody_doublings = []
    bass_inst = None
    bass_doublings = []
    harmony_insts = []
    counter_inst = None

    for name, role in roles.items():
        norm = normalize_instrument_name(name)
        if role == "melody" and melody_inst is None:
            melody_inst = norm
        elif role == "doubling":
            melody_doublings.append(norm)
        elif role == "bass" and bass_inst is None:
            bass_inst = norm
        elif role == "countermelody" and counter_inst is None:
            counter_inst = norm
        elif role == "harmony":
            harmony_insts.append(norm)
        elif role == "percussion":
            pass  # skip for JSON plan

    if not melody_inst:
        return None

    # Build advice from actual score properties
    family_desc = ", ".join(
        f"{fam} ({', '.join(insts[:3])})" for fam, insts in families.items()
        if fam not in ("other", "keyboard")
    )
    advice = (
        f"This {key} piece uses {len(tracks)} instruments across {family_desc}. "
        f"The {melody_inst} carries the melody in the {mel_register} register"
    )
    if melody_doublings:
        advice += f", doubled by {', '.join(melody_doublings[:2])}"
    advice += f". {bass_inst or 'cello'} anchors the bass."
    if counter_inst:
        advice += f" {counter_inst} provides countermelody."

    plan = {
        "melody": {
            "primary": melody_inst,
            "doublings": melody_doublings[:3],
        },
        "bass": {
            "primary": bass_inst or "cello",
            "doublings": bass_doublings[:2],
        },
        "harmony": harmony_insts[:6],
        "countermelody": counter_inst,
        "advice": advice,
    }

    return {
        "instruction": "Create an orchestration plan for this piano piece.",
        "input": input_text,
        "output": json.dumps(plan, indent=2),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate training pairs from SymphonyNet")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--min-tracks", type=int, default=3, help="Min instrument tracks")
    args = parser.parse_args()

    # Gather all MIDI files
    midi_files = []
    for ext in ("*.mid", "*.midi"):
        midi_files.extend(str(p) for p in DATASET_DIR.rglob(ext))
    midi_files.sort()

    if args.limit > 0:
        midi_files = midi_files[:args.limit]

    logger.info(f"Processing {len(midi_files)} MIDI files with {args.workers} workers...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    with open(args.output, "w", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_midi, path): path for path in midi_files}

            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 1000 == 0:
                    logger.info(f"  {i + 1}/{len(midi_files)} processed ({success} pairs)")

                try:
                    result = future.result(timeout=30)
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1

    logger.info(f"Done: {success} training pairs from {len(midi_files)} files ({failed} skipped)")
    logger.info(f"Output: {args.output}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
