#!/usr/bin/env python3
"""Debug orchestration pipeline from the terminal.

Usage:
    cd api
    python scripts/debug_orchestrate.py data/samples/demo.mid --style mahler
    python scripts/debug_orchestrate.py data/samples/demo.mid --style mahler --llm
    python scripts/debug_orchestrate.py data/samples/demo.mid --query "timpani rules"
    python scripts/debug_orchestrate.py data/samples/demo.mid --compare
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decimus.analyzer import analyze_piano
from decimus.planner import create_plan
from decimus.orchestrator import orchestrate_direct
from decimus.knowledge import query_rules


def analyze(midi_path):
    """Analyze and print musical structure."""
    analysis = analyze_piano(midi_path)
    print(f"\n  Key: {analysis.key}")
    print(f"  Tempo: {analysis.tempo:.0f} BPM")
    print(f"  Time: {analysis.time_sig[0]}/{analysis.time_sig[1]}")
    print(f"  Measures: {analysis.total_measures}")
    print(f"  Melody: {len(analysis.melody_notes)} notes")
    print(f"  Bass: {len(analysis.bass_notes)} notes")
    print(f"  Inner: {len(analysis.inner_notes)} notes")
    print(f"  Chords: {len(analysis.chords)}")

    if analysis.chords:
        labels = [c.label for c in analysis.chords[:16] if c.label != "NA"]
        print(f"  Progression: {' - '.join(labels)}")

    return analysis


def orchestrate(analysis, style, ensemble, use_llm=False, output=None):
    """Run orchestration and print results."""
    t0 = time.time()
    plan = create_plan(analysis, style_name=style, ensemble_name=ensemble, use_llm=use_llm)
    t_plan = time.time() - t0

    llm_used = any("[llm]" in a for a in plan.kb_advice)
    method = "LLM" if llm_used else "rule-based"
    print(f"\n  Plan ({method}, {t_plan:.1f}s):")
    for r in plan.roles:
        print(f"    {r.spec.display_name:15s} -> {r.role:15s} ({r.spec.family}, vel={r.velocity_scale:.2f})")

    if output is None:
        output = f"/tmp/decimus_{style}_{ensemble}.mid"

    t0 = time.time()
    result = orchestrate_direct(analysis, plan, output)
    t_orch = time.time() - t0

    print(f"\n  Output ({t_orch:.2f}s): {output}")
    print(f"  Tracks: {result['num_tracks']}, Notes: {result['total_notes']}")
    for name, count in sorted(result['tracks'].items(), key=lambda x: -x[1]):
        bar = "#" * min(40, count)
        print(f"    {name:15s}: {count:3d}  {bar}")

    # Render if fluidsynth available
    try:
        from decimus.renderer import render_midi_to_wav, is_available
        if is_available():
            wav = output.replace(".mid", ".wav")
            render_midi_to_wav(output, wav)
            size = os.path.getsize(wav) / 1024 / 1024
            print(f"\n  Audio: {wav} ({size:.1f} MB)")
            print(f"  Play:  open {wav}")
    except Exception as e:
        print(f"\n  Audio rendering skipped: {e}")

    if plan.kb_advice:
        print(f"\n  KB Advice ({len(plan.kb_advice)} entries):")
        for a in plan.kb_advice[:3]:
            print(f"    {a[:100]}")

    return plan, result


def query_kb(question, n=5):
    """Query the knowledge base."""
    print(f"\n  Query: \"{question}\"\n")
    results = query_rules(question, n_results=n)
    for i, r in enumerate(results):
        print(f"  [{i+1}] relevance={r.relevance:.2f}")
        # Print full text, word-wrapped
        text = r.text.strip()
        for line in text.split('\n')[:10]:
            print(f"      {line}")
        print()


def compare(analysis, styles, ensemble):
    """Compare multiple styles side by side."""
    print(f"\n  {'Style':12s} | {'Tracks':6s} | {'Notes':5s} | {'Strings':7s} | {'Winds':5s} | {'Brass':5s} | {'Timp':4s} | Time")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-{'-'*6}")

    for style in styles:
        t0 = time.time()
        plan = create_plan(analysis, style_name=style, ensemble_name=ensemble, use_llm=False)
        result = orchestrate_direct(analysis, plan, f"/tmp/dec_{style}.mid")
        elapsed = time.time() - t0

        tr = result['tracks']
        strings = sum(tr.get(s, 0) for s in ['Violin I', 'Violin II', 'Viola'])
        winds = sum(tr.get(s, 0) for s in ['Flute', 'Clarinet', 'Oboe', 'Bassoon'])
        brass = sum(tr.get(s, 0) for s in ['Trumpet', 'Trombone', 'Tuba', 'Horn'])
        timp = tr.get('Timpani', 0)

        print(f"  {style:12s} | {result['num_tracks']:6d} | {result['total_notes']:5d} | {strings:7d} | {winds:5d} | {brass:5d} | {timp:4d} | {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Debug Decimus orchestration pipeline")
    parser.add_argument("midi", help="Input MIDI file")
    parser.add_argument("--style", "-s", default="mahler", help="Composer style")
    parser.add_argument("--ensemble", "-e", default="full", help="Ensemble type")
    parser.add_argument("--llm", action="store_true", help="Use LLM for planning")
    parser.add_argument("--output", "-o", help="Output MIDI path")
    parser.add_argument("--query", "-q", help="Query knowledge base instead of orchestrating")
    parser.add_argument("--compare", action="store_true", help="Compare all styles")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't orchestrate")
    args = parser.parse_args()

    print(f"\n  Decimus Debug Tool")
    print(f"  {'='*40}")

    if args.query:
        query_kb(args.query)
        return

    analysis = analyze(args.midi)

    if args.analyze_only:
        return

    if args.compare:
        from decimus.styles import STYLES
        compare(analysis, list(STYLES.keys()), args.ensemble)
        return

    orchestrate(analysis, args.style, args.ensemble, use_llm=args.llm, output=args.output)


if __name__ == "__main__":
    main()
