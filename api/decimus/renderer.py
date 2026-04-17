"""MIDI-to-audio rendering using FluidSynth + orchestral SoundFont.

Converts generated MIDI files to WAV/MP3 for instant playback in the web UI
without requiring a DAW.

Requirements:
    brew install fluid-synth
    pip install midi2audio
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SF_DIR = Path(__file__).resolve().parents[1] / "data" / "soundfonts"

# Preferred SoundFonts in priority order (orchestral quality first)
_PREFERRED = ["Sonatina_Symphonic_Orchestra.sf2", "FluidR3_GM.sf2"]


def _find_default_soundfont() -> Path:
    for name in _PREFERRED:
        candidate = _SF_DIR / name
        if candidate.exists():
            return candidate
    # Fall back to any .sf2 the user dropped in
    available = sorted(_SF_DIR.glob("*.sf2"))
    if available:
        return available[0]
    # No soundfont present — return the expected path so callers can check .exists()
    return _SF_DIR / _PREFERRED[-1]


DEFAULT_SF2 = _find_default_soundfont()


def render_midi_to_wav(
    midi_path: str,
    wav_path: str | None = None,
    soundfont: str | None = None,
    sample_rate: int = 44100,
    gain: float = 0.6,
) -> str:
    """Render a MIDI file to WAV using FluidSynth.

    Args:
        midi_path: Path to input MIDI file.
        wav_path: Output WAV path. Defaults to same name with .wav extension.
        soundfont: Path to .sf2 file. Defaults to FluidR3_GM.sf2.
        sample_rate: Audio sample rate (default 44100).
        gain: Master gain 0.0-1.0 (default 0.6).

    Returns:
        Path to the rendered WAV file.
    """
    sf2 = soundfont or str(DEFAULT_SF2)
    if not os.path.exists(sf2):
        raise FileNotFoundError(
            f"SoundFont not found: {sf2}\n"
            f"Drop any .sf2 file into {_SF_DIR} to enable audio rendering. "
            f"See the README section 'SoundFonts' for download links."
        )

    if wav_path is None:
        wav_path = str(Path(midi_path).with_suffix(".wav"))

    logger.info(f"Rendering MIDI → WAV: {Path(midi_path).name} (sr={sample_rate}, gain={gain})")

    # Use fluidsynth CLI for reliable rendering
    cmd = [
        "fluidsynth",
        "-ni",              # no interactive, no MIDI input
        "-g", str(gain),    # master gain
        "-r", str(sample_rate),
        "-F", wav_path,     # render to file
        sf2,
        midi_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"FluidSynth error: {result.stderr}")
            raise RuntimeError(f"FluidSynth failed: {result.stderr[:500]}")
    except FileNotFoundError:
        raise RuntimeError(
            "FluidSynth not found. Install with: brew install fluid-synth"
        )

    if not os.path.exists(wav_path):
        raise RuntimeError("FluidSynth produced no output file")

    size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    logger.info(f"Rendered WAV: {Path(wav_path).name} ({size_mb:.1f} MB)")
    return wav_path


def render_midi_to_mp3(
    midi_path: str,
    mp3_path: str | None = None,
    soundfont: str | None = None,
    sample_rate: int = 44100,
    bitrate: str = "192k",
    gain: float = 0.6,
) -> str:
    """Render MIDI to MP3 via FluidSynth → ffmpeg.

    Args:
        midi_path: Path to input MIDI file.
        mp3_path: Output MP3 path. Defaults to same name with .mp3 extension.
        soundfont: Path to .sf2 file.
        sample_rate: Audio sample rate.
        bitrate: MP3 bitrate (default 192k).
        gain: Master gain 0.0-1.0.

    Returns:
        Path to the rendered MP3 file.
    """
    if mp3_path is None:
        mp3_path = str(Path(midi_path).with_suffix(".mp3"))

    # First render to WAV
    wav_path = mp3_path.replace(".mp3", ".tmp.wav")
    render_midi_to_wav(midi_path, wav_path, soundfont, sample_rate, gain)

    # Convert WAV → MP3 with ffmpeg
    logger.info(f"Converting WAV → MP3: {bitrate}")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-b:a", bitrate,
            "-q:a", "2",
            mp3_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr[:500]}")
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
    except FileNotFoundError:
        # No ffmpeg — fall back to WAV
        logger.warning("ffmpeg not found, keeping WAV format")
        os.rename(wav_path, mp3_path.replace(".mp3", ".wav"))
        return mp3_path.replace(".mp3", ".wav")
    finally:
        # Clean up temp WAV
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
    logger.info(f"Rendered MP3: {Path(mp3_path).name} ({size_mb:.1f} MB)")
    return mp3_path


def is_available() -> bool:
    """Check if FluidSynth and SoundFont are available."""
    try:
        result = subprocess.run(
            ["fluidsynth", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    return DEFAULT_SF2.exists()
