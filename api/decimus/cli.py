"""Decimus CLI — AI Orchestration Engine.

Transform piano MIDI sketches into full symphonic scores.
"""

import os
import time

import click

# Ensure project root is findable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _print_header():
    """Print the Decimus header."""
    from . import __version__
    click.echo()
    click.echo(click.style(f"  Decimus v{__version__}", fg="cyan", bold=True) +
               click.style(" -- AI Orchestration Engine", fg="white"))
    click.echo()


def _print_step(num: int, total: int, msg: str):
    """Print a pipeline step."""
    click.echo(click.style(f"  [{num}/{total}] ", fg="cyan") + msg)


def _print_detail(msg: str):
    """Print an indented detail line."""
    click.echo(click.style(f"        {msg}", fg="white", dim=True))


@click.group()
def main():
    """Decimus -- AI Orchestration Engine.

    Transform piano MIDI sketches into full symphonic scores.
    """
    pass


@main.command()
@click.argument("midi_path", type=click.Path(exists=True))
@click.option("--style", type=click.Choice(["mozart", "beethoven", "tchaikovsky", "brahms", "mahler",
                                            "debussy", "ravel", "stravinsky", "williams", "zimmer"]),
              default="tchaikovsky", help="Composer style preset.")
@click.option("--ensemble", type=click.Choice(["full", "strings", "chamber", "winds"]),
              default="full", help="Ensemble type.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output MIDI file path.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output.")
@click.option("--no-llm", is_flag=True, help="Disable Decimus LLM, use rules only.")
@click.option("--llm-url", envvar="DECIMUS_LLM_URL", default=None,
              help="Decimus LLM endpoint URL (or set DECIMUS_LLM_URL env var).")
def orchestrate(midi_path, style, ensemble, output, verbose, no_llm, llm_url):
    """Transform a piano MIDI into a full orchestral score.

    Preserves your original melody, harmony, and bass — distributes them
    across orchestral instruments based on the chosen style.

    Use --no-llm to force rule-based planning (no LLM queries).
    """
    _print_header()
    start_time = time.time()

    midi_path = os.path.abspath(midi_path)

    # Configure LLM endpoint if provided
    if llm_url:
        os.environ["DECIMUS_LLM_URL"] = llm_url

    # Step 1: Analyze
    _print_step(1, 4, "Analyzing piano input...")
    from .analyzer import analyze_piano
    analysis = analyze_piano(midi_path)
    _print_detail(f"Key: {analysis.key} | Time: {analysis.time_sig[0]}/{analysis.time_sig[1]} | "
                  f"Tempo: {analysis.tempo:.0f} BPM")
    _print_detail(f"Measures: {analysis.total_measures} | "
                  f"Melody notes: {len(analysis.melody_notes)} | "
                  f"Bass notes: {len(analysis.bass_notes)} | "
                  f"Inner voices: {len(analysis.inner_notes)}")

    # Step 2: Plan
    use_llm = not no_llm
    plan_label = "Planning orchestration"
    if use_llm:
        plan_label += " (Decimus LLM)..."
    else:
        plan_label += " (rule-based)..."
    _print_step(2, 4, plan_label)
    from .planner import create_plan
    plan = create_plan(analysis, style_name=style, ensemble_name=ensemble,
                       use_llm=use_llm)

    llm_used = any("[llm]" in a for a in plan.kb_advice)
    if use_llm and llm_used:
        _print_detail("Planning: Decimus LLM-assisted")
    else:
        _print_detail("Planning: rule-based")
    _print_detail(f"Style: {style.capitalize()} | Ensemble: {ensemble.capitalize()} "
                  f"({len(plan.roles)} parts)")
    if verbose:
        for role in plan.roles:
            _print_detail(f"  {role.spec.display_name:15s} -> {role.role}")

    # Step 3: Orchestrate
    _print_step(3, 4, "Distributing voices across instruments...")
    from .engine import DecimusEngine
    engine = DecimusEngine()
    summary = engine.orchestrate(
        midi_path=midi_path,
        style=style,
        ensemble=ensemble,
        output_path=output,
        use_llm=use_llm,
    )

    # Step 4: Done
    _print_step(4, 4, "Writing output...")
    _print_detail(f"Output: {summary['output_path']}")
    _print_detail(f"Parts: {summary['num_tracks']} tracks, "
                  f"{summary['total_notes']} notes")

    if verbose:
        click.echo()
        click.echo(click.style("  Track breakdown:", fg="white", dim=True))
        for name, count in summary["tracks"].items():
            _print_detail(f"  {name:15s}: {count} notes")

        # Show knowledge base advice if available
        kb_advice = summary.get("kb_advice", [])
        if kb_advice:
            click.echo()
            click.echo(click.style("  Knowledge base insights:", fg="yellow", dim=True))
            for advice in kb_advice[:3]:
                _print_detail(f"  {advice}")

    elapsed = time.time() - start_time
    click.echo()
    click.echo(click.style(f"  Done in {elapsed:.1f}s", fg="green", bold=True))
    click.echo()


@main.command()
@click.argument("midi_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis.")
def analyze(midi_path, verbose):
    """Analyze a piano MIDI file and show its musical structure."""
    _print_header()

    midi_path = os.path.abspath(midi_path)

    click.echo(click.style("  Analyzing: ", fg="cyan") + os.path.basename(midi_path))
    click.echo()

    from .analyzer import analyze_piano
    analysis = analyze_piano(midi_path)

    click.echo(f"  Key:            {analysis.key}")
    click.echo(f"  Time Signature: {analysis.time_sig[0]}/{analysis.time_sig[1]}")
    click.echo(f"  Tempo:          {analysis.tempo:.0f} BPM")
    click.echo(f"  Measures:       {analysis.total_measures}")
    click.echo()
    click.echo(f"  Melody notes:   {len(analysis.melody_notes)}")
    click.echo(f"  Bass notes:     {len(analysis.bass_notes)}")
    click.echo(f"  Inner notes:    {len(analysis.inner_notes)}")
    click.echo(f"  Chords:         {len(analysis.chords)}")
    click.echo(f"  Phrases:        {len(analysis.phrase_boundaries)}")

    if verbose and analysis.chords:
        click.echo()
        click.echo(click.style("  Chord progression:", fg="white", dim=True))
        chords_display = []
        for chord in analysis.chords[:32]:  # Show first 32 measures
            if chord.label != "NA":
                chords_display.append(chord.label.replace("H", ""))
            else:
                chords_display.append("-")
        # Print in groups of 4 (typical phrase)
        for i in range(0, len(chords_display), 4):
            group = chords_display[i:i+4]
            _print_detail(f"  m{i+1:3d}: " + " | ".join(f"{c:6s}" for c in group))

    if verbose and analysis.phrase_boundaries:
        click.echo()
        click.echo(click.style("  Phrase boundaries:", fg="white", dim=True))
        _print_detail("  Measures: " + ", ".join(str(m + 1) for m in analysis.phrase_boundaries))

    click.echo()


@main.command()
@click.argument("midi_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output MIDI file path.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output.")
def reharmonize(midi_path, output, verbose):
    """Re-harmonize a piano MIDI melody for string orchestra.

    Extracts the melody, generates new chord voicings in the detected key,
    and writes a 5-part string arrangement (Vn I, Vn II, Vla, Vc, Cb)
    following Rimsky-Korsakov's orchestration principles.
    """
    _print_header()
    start_time = time.time()

    midi_path = os.path.abspath(midi_path)

    # Step 1: Analyze
    _print_step(1, 3, "Analyzing piano input...")
    from .analyzer import analyze_piano
    analysis = analyze_piano(midi_path)
    _print_detail(f"Key: {analysis.key} | Time: {analysis.time_sig[0]}/{analysis.time_sig[1]} | "
                  f"Tempo: {analysis.tempo:.0f} BPM")
    _print_detail(f"Melody: {len(analysis.melody_notes)} notes")

    if verbose:
        names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        mel_str = " -> ".join(
            f"{names[n.pitch % 12]}{n.pitch // 12 - 1}"
            for n in analysis.melody_notes
        )
        _print_detail(f"  {mel_str}")

    # Step 2: Re-harmonize
    _print_step(2, 3, "Re-harmonizing for string orchestra...")
    from .harmonizer import harmonize_melody
    result = harmonize_melody(analysis, output_path=output)

    # Step 3: Done
    _print_step(3, 3, "Writing output...")
    _print_detail(f"Output: {result['output_path']}")
    _print_detail(f"Parts: {result['num_tracks']} tracks, "
                  f"{result['total_notes']} notes")

    if verbose:
        click.echo()
        click.echo(click.style("  Chord progression:", fg="yellow"))
        chords = result.get("chords", [])
        # Print in groups of 4
        for i in range(0, len(chords), 4):
            group = chords[i:i+4]
            _print_detail("  " + " | ".join(f"{c:10s}" for c in group))

        click.echo()
        click.echo(click.style("  Track breakdown:", fg="white", dim=True))
        for name, count in result["tracks"].items():
            _print_detail(f"  {name:15s}: {count} notes")

    elapsed = time.time() - start_time
    click.echo()
    click.echo(click.style(f"  Done in {elapsed:.1f}s", fg="green", bold=True))
    click.echo()


if __name__ == "__main__":
    main()
