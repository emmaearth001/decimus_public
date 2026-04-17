"""Orchestral instrument database with GM programs, playable ranges, and families."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    display_name: str
    program: int  # General MIDI program number
    low: int      # Lowest playable MIDI note
    high: int     # Highest playable MIDI note
    family: str   # strings, woodwinds, brass, percussion, keyboard
    is_drum: bool = False
    is_monophonic: bool = False  # True for instruments that can only play one note at a time


# Full orchestral instrument database
INSTRUMENTS = {
    # Strings
    "violin_1":   InstrumentSpec("violin_1",   "Violin I",    40, 55, 105, "strings"),
    "violin_2":   InstrumentSpec("violin_2",   "Violin II",   40, 55, 105, "strings"),
    "viola":      InstrumentSpec("viola",       "Viola",       41, 48,  93, "strings"),
    "cello":      InstrumentSpec("cello",       "Cello",       42, 36,  76, "strings"),
    "contrabass": InstrumentSpec("contrabass",  "Contrabass",  43, 28,  60, "strings"),

    # Woodwinds (monophonic)
    "flute":      InstrumentSpec("flute",       "Flute",       73, 60,  96, "woodwinds", is_monophonic=True),
    "oboe":       InstrumentSpec("oboe",        "Oboe",        68, 58,  91, "woodwinds", is_monophonic=True),
    "clarinet":   InstrumentSpec("clarinet",    "Clarinet",    71, 50,  94, "woodwinds", is_monophonic=True),
    "bassoon":    InstrumentSpec("bassoon",     "Bassoon",     70, 34,  75, "woodwinds", is_monophonic=True),

    # Brass (monophonic)
    "horn":       InstrumentSpec("horn",        "Horn",        60, 34,  77, "brass", is_monophonic=True),
    "trumpet":    InstrumentSpec("trumpet",     "Trumpet",     56, 55,  82, "brass", is_monophonic=True),
    "trombone":   InstrumentSpec("trombone",    "Trombone",    57, 34,  72, "brass", is_monophonic=True),
    "tuba":       InstrumentSpec("tuba",        "Tuba",        58, 24,  60, "brass", is_monophonic=True),

    # Percussion (monophonic — one pitch at a time)
    "timpani":    InstrumentSpec("timpani",     "Timpani",     47, 40,  55, "percussion", is_monophonic=True),

    # GM Drum Kit — is_drum=True routes to channel 10
    # Pitch values are GM drum map: 35=kick, 38=snare, 42=hi-hat, 49=crash, 51=ride
    "drums":      InstrumentSpec("drums",       "Percussion",   0, 35,  81, "percussion", is_drum=True),

    # Other
    "harp":       InstrumentSpec("harp",        "Harp",        46, 24, 103, "strings"),
    "piano":      InstrumentSpec("piano",       "Piano",        0, 21, 108, "keyboard"),
}


# Ensemble configurations: which instruments are included
ENSEMBLES = {
    "full": [
        "violin_1", "violin_2", "viola", "cello", "contrabass",
        "flute", "oboe", "clarinet", "bassoon",
        "horn", "trumpet", "trombone", "tuba",
        "timpani", "drums", "harp",
    ],
    "strings": [
        "violin_1", "violin_2", "viola", "cello", "contrabass",
    ],
    "chamber": [
        "violin_1", "violin_2", "viola", "cello",
        "flute", "oboe", "clarinet", "bassoon", "horn", "piano",
    ],
    "winds": [
        "flute", "oboe", "clarinet", "bassoon",
        "horn", "trumpet", "trombone",
    ],
}


def get_ensemble(name: str) -> list[InstrumentSpec]:
    """Return list of InstrumentSpec for a named ensemble."""
    keys = ENSEMBLES.get(name)
    if keys is None:
        raise ValueError(f"Unknown ensemble: {name!r}. Choose from: {list(ENSEMBLES.keys())}")
    return [INSTRUMENTS[k] for k in keys]


def clamp_to_range(pitch: int, spec: InstrumentSpec) -> int:
    """Clamp a MIDI pitch into an instrument's playable range via octave transposition."""
    while pitch < spec.low and pitch + 12 <= spec.high:
        pitch += 12
    while pitch > spec.high and pitch - 12 >= spec.low:
        pitch -= 12
    return pitch


def in_range(pitch: int, spec: InstrumentSpec) -> bool:
    """Check if a MIDI pitch is within an instrument's playable range."""
    return spec.low <= pitch <= spec.high
