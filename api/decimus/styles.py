"""Orchestration style presets — one per composer with distinct orchestration DNA."""

from dataclasses import dataclass, field


@dataclass
class StylePreset:
    name: str
    description: str  # composer full name for display

    # Instruments prioritized for each musical role (ordered by preference)
    melody_instruments: list[str]
    harmony_instruments: list[str]
    bass_instruments: list[str]
    countermelody_instruments: list[str]

    # Which melody instruments are doubled (play the same notes)
    melody_doublings: list[str] = field(default_factory=list)
    bass_doublings: list[str] = field(default_factory=list)

    # Percussion style: None, "sparse", "rhythmic", "dramatic"
    percussion_style: str | None = None

    # Density control
    max_simultaneous_voices: int = 12
    density_target: float = 4.0

    # Instrument bias strength for constrained generation
    instrument_bias: float = 2.0

    # Temperature adjustments
    event_temperature: float = 1.0
    instrument_temperature: float = 0.7


STYLES = {
    "mozart": StylePreset(
        name="mozart",
        description="Mozart",
        melody_instruments=["violin_1", "oboe", "flute"],
        melody_doublings=[],
        harmony_instruments=["violin_2", "viola"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["viola", "bassoon"],
        percussion_style=None,
        max_simultaneous_voices=8,
        density_target=3.0,
        instrument_bias=2.5,
        event_temperature=0.9,
    ),
    "beethoven": StylePreset(
        name="beethoven",
        description="Beethoven",
        melody_instruments=["violin_1", "oboe", "clarinet"],
        melody_doublings=["flute"],
        harmony_instruments=["violin_2", "viola", "horn", "clarinet"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["cello", "bassoon", "horn"],
        percussion_style="sparse",
        max_simultaneous_voices=12,
        density_target=4.0,
        instrument_bias=2.0,
        event_temperature=1.0,
    ),
    "tchaikovsky": StylePreset(
        name="tchaikovsky",
        description="Tchaikovsky",
        melody_instruments=["violin_1", "flute", "oboe"],
        melody_doublings=["flute"],
        harmony_instruments=["violin_2", "viola", "clarinet", "horn"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["cello", "horn", "clarinet"],
        percussion_style="dramatic",
        max_simultaneous_voices=15,
        density_target=5.0,
        instrument_bias=2.0,
        event_temperature=1.0,
    ),
    "brahms": StylePreset(
        name="brahms",
        description="Brahms",
        melody_instruments=["violin_1", "clarinet", "horn"],
        melody_doublings=[],
        harmony_instruments=["violin_2", "viola", "horn", "bassoon"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["viola", "cello", "horn"],
        percussion_style=None,
        max_simultaneous_voices=12,
        density_target=4.5,
        instrument_bias=2.0,
        event_temperature=0.95,
        instrument_temperature=0.65,
    ),
    "mahler": StylePreset(
        name="mahler",
        description="Mahler",
        melody_instruments=["oboe", "violin_1", "clarinet", "horn"],
        melody_doublings=[],
        harmony_instruments=["horn", "viola", "clarinet", "bassoon"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["cello", "horn", "oboe", "bassoon"],
        percussion_style="dramatic",
        max_simultaneous_voices=14,
        density_target=4.0,
        instrument_bias=2.0,
        event_temperature=0.8,
        instrument_temperature=0.6,
    ),
    "debussy": StylePreset(
        name="debussy",
        description="Debussy",
        melody_instruments=["flute", "oboe", "clarinet"],
        melody_doublings=[],
        harmony_instruments=["viola", "violin_2", "horn", "harp"],
        bass_instruments=["cello", "contrabass"],
        bass_doublings=[],
        countermelody_instruments=["flute", "clarinet", "oboe"],
        percussion_style=None,
        max_simultaneous_voices=10,
        density_target=3.5,
        instrument_bias=1.8,
        event_temperature=1.0,
        instrument_temperature=0.7,
    ),
    "ravel": StylePreset(
        name="ravel",
        description="Ravel",
        melody_instruments=["flute", "oboe", "violin_1", "clarinet"],
        melody_doublings=["clarinet"],
        harmony_instruments=["violin_2", "viola", "horn", "harp"],
        bass_instruments=["cello", "contrabass", "bassoon"],
        bass_doublings=["contrabass"],
        countermelody_instruments=["horn", "cello", "oboe"],
        percussion_style="sparse",
        max_simultaneous_voices=13,
        density_target=4.5,
        instrument_bias=1.6,
        event_temperature=1.0,
        instrument_temperature=0.65,
    ),
    "stravinsky": StylePreset(
        name="stravinsky",
        description="Stravinsky",
        melody_instruments=["clarinet", "trumpet", "violin_1"],
        melody_doublings=[],
        harmony_instruments=["horn", "viola", "oboe", "trombone"],
        bass_instruments=["contrabass", "tuba", "bassoon"],
        bass_doublings=["tuba"],
        countermelody_instruments=["trumpet", "flute", "cello"],
        percussion_style="rhythmic",
        max_simultaneous_voices=12,
        density_target=4.0,
        instrument_bias=1.5,
        event_temperature=1.1,
    ),
    "williams": StylePreset(
        name="williams",
        description="John Williams",
        melody_instruments=["violin_1", "horn", "trumpet"],
        melody_doublings=["horn", "flute"],
        harmony_instruments=["violin_2", "viola", "clarinet", "trombone"],
        bass_instruments=["cello", "contrabass", "tuba"],
        bass_doublings=["contrabass", "tuba"],
        countermelody_instruments=["cello", "horn", "oboe"],
        percussion_style="dramatic",
        max_simultaneous_voices=18,
        density_target=6.0,
        instrument_bias=1.8,
        event_temperature=1.0,
    ),
    "zimmer": StylePreset(
        name="zimmer",
        description="Hans Zimmer",
        melody_instruments=["violin_1", "cello", "horn"],
        melody_doublings=["cello"],
        harmony_instruments=["violin_2", "viola", "trombone", "horn"],
        bass_instruments=["contrabass", "tuba", "cello"],
        bass_doublings=["tuba", "contrabass"],
        countermelody_instruments=["horn", "trumpet", "cello"],
        percussion_style="rhythmic",
        max_simultaneous_voices=16,
        density_target=5.5,
        instrument_bias=2.0,
        event_temperature=1.0,
        instrument_temperature=0.8,
    ),
}


def get_style(name: str) -> StylePreset:
    """Return a StylePreset by name."""
    preset = STYLES.get(name)
    if preset is None:
        raise ValueError(f"Unknown style: {name!r}. Choose from: {list(STYLES.keys())}")
    return preset
