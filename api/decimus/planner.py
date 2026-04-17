"""Orchestration planner: assigns instrument roles based on piano analysis, style,
and Rimsky-Korsakov orchestration principles from the knowledge base.

When use_llm is True, queries the fine-tuned Decimus LLM for instrument
assignments before falling back to rule-based logic.
"""

import logging
from dataclasses import dataclass, field

from .analyzer import PianoAnalysis
from .instruments import INSTRUMENTS, InstrumentSpec, get_ensemble, in_range
from .styles import StylePreset, get_style

logger = logging.getLogger(__name__)


@dataclass
class InstrumentRole:
    spec: InstrumentSpec
    role: str              # "melody", "harmony", "bass", "countermelody", "doubling"
    track_id: int          # Track index for SymphonyNet (0-39)
    doubles: str | None    # Name of instrument this doubles, or None
    velocity_scale: float  # Velocity multiplier (0.0-1.0)


@dataclass
class OrchestrationPlan:
    style: StylePreset
    ensemble_name: str
    roles: list[InstrumentRole] = field(default_factory=list)
    track_to_instrument: dict[int, InstrumentSpec] = field(default_factory=dict)
    instrument_programs: set[int] = field(default_factory=set)
    kb_advice: list[str] = field(default_factory=list)  # knowledge base advice used

    @property
    def melody_tracks(self) -> list[int]:
        return [r.track_id for r in self.roles if r.role in ("melody", "doubling")]

    @property
    def bass_tracks(self) -> list[int]:
        return [r.track_id for r in self.roles if r.role == "bass"]

    @property
    def harmony_tracks(self) -> list[int]:
        return [r.track_id for r in self.roles if r.role == "harmony"]

    @property
    def active_tracks(self) -> list[int]:
        return [r.track_id for r in self.roles]

    @property
    def max_tracks(self) -> int:
        return len(self.roles)


def create_plan(
    analysis: PianoAnalysis,
    style_name: str = "tchaikovsky",
    ensemble_name: str = "full",
    use_knowledge_base: bool = True,
    use_llm: bool = True,
) -> OrchestrationPlan:
    """Create an orchestration plan from piano analysis and style/ensemble choices.

    When use_llm is True, queries the fine-tuned Decimus LLM for planning.
    Falls back to rule-based logic if the LLM is unavailable or returns
    an invalid plan.

    When use_knowledge_base is True, queries Rimsky-Korsakov's Principles of
    Orchestration to refine instrument selection and doublings.
    """
    style = get_style(style_name)
    ensemble = get_ensemble(ensemble_name)

    # Try LLM-based planning first
    if use_llm:
        llm_plan = _try_llm_plan(analysis, style, ensemble, style_name, ensemble_name)
        if llm_plan is not None:
            return llm_plan

    plan = OrchestrationPlan(
        style=style,
        ensemble_name=ensemble_name,
    )

    # Determine melody/bass register to guide instrument choice
    melody_register = _compute_register(analysis.melody_notes)
    bass_register = _compute_register(analysis.bass_notes)

    # Optionally query the knowledge base for instrument advice
    kb_melody_advice = []
    kb_doubling_advice = []
    if use_knowledge_base:
        try:
            from .knowledge import (
                get_harmony_advice,
                get_melody_doubling_advice,
                get_style_advice,
            )
            # Get style-specific advice
            style_advice = get_style_advice(style_name)
            for a in style_advice[:2]:
                plan.kb_advice.append(f"[style] {a.text[:200]}")

            # Get melody doubling advice based on the primary melody instrument
            primary_inst = style.melody_instruments[0] if style.melody_instruments else "violin"
            kb_doubling_advice = get_melody_doubling_advice(analysis.key, primary_inst)

            # Get harmony advice
            density = "sparse" if len(analysis.inner_notes) < 20 else "moderate" if len(analysis.inner_notes) < 100 else "dense"
            harmony_advice = get_harmony_advice(analysis.key, density)
            for a in harmony_advice[:2]:
                plan.kb_advice.append(f"[harmony] {a.text[:200]}")

        except Exception as e:
            logger.debug(f"Knowledge base query failed: {e}")

    # Reorder melody instruments based on register fitness
    melody_candidates = _rank_by_register(
        style.melody_instruments, melody_register, ensemble
    )

    track_id = 4  # Start at 4 (0-3 reserved in SymphonyNet)

    # 1. Assign primary melody instrument (best register fit)
    for inst_name in melody_candidates:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        # Verify the melody actually fits this instrument's range
        if analysis.melody_notes and not _notes_fit_instrument(analysis.melody_notes, spec):
            continue
        role = InstrumentRole(
            spec=spec,
            role="melody",
            track_id=track_id,
            doubles=None,
            velocity_scale=1.0,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1
        break

    # Melody doublings — filter to instruments that can cover the melody range
    for inst_name in style.melody_doublings:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        if analysis.melody_notes and not _notes_fit_instrument(analysis.melody_notes, spec):
            continue
        primary_melody = style.melody_instruments[0]
        role = InstrumentRole(
            spec=spec,
            role="doubling",
            track_id=track_id,
            doubles=primary_melody,
            velocity_scale=0.8,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1

    # 2. Assign bass instruments
    bass_candidates = _rank_by_register(
        style.bass_instruments, bass_register, ensemble
    )
    for inst_name in bass_candidates:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        role = InstrumentRole(
            spec=spec,
            role="bass",
            track_id=track_id,
            doubles=None,
            velocity_scale=0.9,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1

    # Bass doublings (skip if already assigned)
    for inst_name in style.bass_doublings:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        if any(r.spec.name == inst_name for r in plan.roles):
            continue
        role = InstrumentRole(
            spec=spec,
            role="bass",
            track_id=track_id,
            doubles=style.bass_instruments[0],
            velocity_scale=0.85,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1

    # 3. Assign countermelody instrument (before harmony so it gets reserved)
    for inst_name in style.countermelody_instruments:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        if any(r.spec.name == inst_name for r in plan.roles):
            continue
        role = InstrumentRole(
            spec=spec,
            role="countermelody",
            track_id=track_id,
            doubles=None,
            velocity_scale=0.85,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1
        break  # One countermelody voice

    # 4. Assign harmony instruments
    for inst_name in style.harmony_instruments:
        spec = INSTRUMENTS.get(inst_name)
        if spec is None or spec not in ensemble:
            continue
        if any(r.spec.name == inst_name for r in plan.roles):
            continue
        role = InstrumentRole(
            spec=spec,
            role="harmony",
            track_id=track_id,
            doubles=None,
            velocity_scale=0.7,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1

    # 5. Assign timpani — dedicated role, NOT harmony
    #    Timpani plays tonic/dominant on strong beats, not random inner voice notes
    for spec in ensemble:
        if spec.name == "timpani":
            if any(r.spec.name == spec.name for r in plan.roles):
                break
            role = InstrumentRole(
                spec=spec,
                role="timpani",
                track_id=track_id,
                doubles=None,
                velocity_scale=0.80,
            )
            plan.roles.append(role)
            plan.track_to_instrument[track_id] = spec
            plan.instrument_programs.add(spec.program)
            track_id += 1
            break

    # 6. Assign drum kit percussion if style uses it
    if style.percussion_style:
        for spec in ensemble:
            if spec.family == "percussion" and spec.is_drum:
                if any(r.spec.name == spec.name for r in plan.roles):
                    continue
                role = InstrumentRole(
                    spec=spec,
                    role="percussion",
                    track_id=track_id,
                    doubles=None,
                    velocity_scale=0.75,
                )
                plan.roles.append(role)
                plan.track_to_instrument[track_id] = spec
                plan.instrument_programs.add(spec.program)
                track_id += 1

    # 7. Fill remaining ensemble instruments as harmony
    for spec in ensemble:
        if any(r.spec.name == spec.name for r in plan.roles):
            continue
        if spec.is_drum or spec.name == "timpani":
            continue  # Don't auto-fill percussion as harmony
        role = InstrumentRole(
            spec=spec,
            role="harmony",
            track_id=track_id,
            doubles=None,
            velocity_scale=0.65,
        )
        plan.roles.append(role)
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1

    return plan


def _compute_register(notes: list) -> str:
    """Classify a note list's register as 'low', 'mid', or 'high'."""
    if not notes:
        return "mid"
    avg_pitch = sum(n.pitch for n in notes) / len(notes)
    if avg_pitch < 55:
        return "low"
    elif avg_pitch > 75:
        return "high"
    return "mid"


def _rank_by_register(
    instrument_names: list[str],
    register: str,
    ensemble: list[InstrumentSpec],
) -> list[str]:
    """Reorder instrument candidates to prefer those matching the register.

    For high register: prefer flute, oboe, violin
    For low register: prefer cello, bassoon, trombone
    For mid register: keep original style order
    """
    if register == "mid":
        return instrument_names

    ensemble_names = {s.name for s in ensemble}
    available = [n for n in instrument_names if n in ensemble_names]

    if register == "high":
        high_pref = {"flute", "oboe", "violin_1", "violin_2", "clarinet", "trumpet"}
        preferred = [n for n in available if n in high_pref]
        rest = [n for n in available if n not in high_pref]
        return preferred + rest
    else:  # low
        low_pref = {"cello", "contrabass", "bassoon", "trombone", "tuba", "horn"}
        preferred = [n for n in available if n in low_pref]
        rest = [n for n in available if n not in low_pref]
        return preferred + rest


def _notes_fit_instrument(notes: list, spec: InstrumentSpec, tolerance: float = 0.8) -> bool:
    """Check if at least `tolerance` fraction of notes fit within instrument range."""
    if not notes:
        return True
    fit_count = sum(1 for n in notes if in_range(n.pitch, spec))
    return (fit_count / len(notes)) >= tolerance


# ---------------------------------------------------------------------------
# Instrument name normalization (for LLM output parsing)
# ---------------------------------------------------------------------------

# Map natural language instrument names to Decimus internal names
_INSTRUMENT_ALIASES = {
    "violin": "violin_1", "violin i": "violin_1", "violin 1": "violin_1",
    "first violin": "violin_1", "1st violin": "violin_1",
    "violin ii": "violin_2", "violin 2": "violin_2",
    "second violin": "violin_2", "2nd violin": "violin_2",
    "solo oboe": "oboe", "solo flute": "flute", "solo clarinet": "clarinet",
    "solo bassoon": "bassoon", "solo trumpet": "trumpet", "solo horn": "horn",
    "solo cello": "cello", "solo violin": "violin_1",
    "english horn": "oboe",  # closest GM mapping
    "cor anglais": "oboe",
    "french horn": "horn",
    "double bass": "contrabass", "string bass": "contrabass",
    "bass trombone": "trombone",
    "muted trumpet": "trumpet", "muted horn": "horn",
}


def _normalize_instrument_name(raw: str) -> str:
    """Extract a Decimus instrument name from an LLM description.

    Handles strings like 'Solo Oboe, espressivo' or 'Muted Trumpet con sordino'.
    """
    # Take text before comma or parenthesis
    name = raw.split(",")[0].split("(")[0].strip().lower()
    # Try direct alias match
    if name in _INSTRUMENT_ALIASES:
        return _INSTRUMENT_ALIASES[name]
    # Try matching against known instrument names
    for inst_name in INSTRUMENTS:
        if inst_name in name or inst_name.replace("_", " ") in name:
            return inst_name
    # Try alias partial match
    for alias, inst_name in _INSTRUMENT_ALIASES.items():
        if alias in name:
            return inst_name
    return name


def _extract_instrument_names(text: str) -> list[str]:
    """Extract instrument names from a free-text doubling description.

    E.g. 'Flute doubling at the octave with muted horn' → ['flute', 'horn']
    """
    found = []
    text_lower = text.lower()
    for inst_name in INSTRUMENTS:
        readable = inst_name.replace("_", " ")
        if readable in text_lower or inst_name in text_lower:
            found.append(inst_name)
    for alias, inst_name in _INSTRUMENT_ALIASES.items():
        if alias in text_lower and inst_name not in found:
            found.append(inst_name)
    return found


# ---------------------------------------------------------------------------
# Decimus LLM integration
# ---------------------------------------------------------------------------

_llm = None


def _get_llm():
    """Lazy-load the Decimus LLM client singleton."""
    global _llm
    if _llm is None:
        from .llm_client import DecimusLLM
        _llm = DecimusLLM()
    return _llm


def _try_llm_plan(
    analysis: PianoAnalysis,
    style: StylePreset,
    ensemble: list[InstrumentSpec],
    style_name: str,
    ensemble_name: str,
) -> OrchestrationPlan | None:
    """Attempt LLM-based planning. Returns None if unavailable or invalid."""
    llm = _get_llm()
    if not llm.is_available():
        return None

    from .llm_client import format_analysis_for_llm

    prompt = format_analysis_for_llm(analysis, style_name, ensemble_name)
    llm_result = llm.query_plan(prompt, style=style_name)
    if llm_result is None:
        return None

    try:
        return _apply_llm_plan(llm_result, style, ensemble, ensemble_name)
    except Exception as e:
        logger.warning(f"LLM plan conversion failed: {e}")
        return None


def _apply_llm_plan(
    llm_plan: dict,
    style: StylePreset,
    ensemble: list[InstrumentSpec],
    ensemble_name: str,
) -> OrchestrationPlan | None:
    """Convert an LLM-generated plan dict into an OrchestrationPlan.

    Validates instrument names against the actual ensemble.
    Returns None if the plan is invalid (missing melody or bass).
    """
    plan = OrchestrationPlan(style=style, ensemble_name=ensemble_name)
    ensemble_names = {s.name for s in ensemble}
    track_id = 4

    used_instruments: set[str] = set()

    def _add_role(inst_name: str, role: str, doubles: str | None,
                  velocity: float) -> bool:
        nonlocal track_id
        if inst_name not in INSTRUMENTS:
            logger.warning(f"LLM suggested unknown instrument: {inst_name!r} for {role}")
            return False
        if inst_name not in ensemble_names:
            logger.info(f"LLM suggested {inst_name!r} for {role} but not in {ensemble_name} ensemble")
            return False
        # Skip duplicates — each instrument should only appear once
        if inst_name in used_instruments:
            logger.debug(f"LLM duplicate skipped: {inst_name!r} already assigned, ignoring {role}")
            return False
        used_instruments.add(inst_name)
        spec = INSTRUMENTS[inst_name]
        plan.roles.append(InstrumentRole(
            spec=spec, role=role, track_id=track_id,
            doubles=doubles, velocity_scale=velocity,
        ))
        plan.track_to_instrument[track_id] = spec
        plan.instrument_programs.add(spec.program)
        track_id += 1
        return True

    # Melody — support both standard format and Mahler's Conductor format
    melody_data = llm_plan.get("melody", {})
    primary = melody_data.get("primary")
    # Mahler Conductor format: primary_instrument is a top-level key
    if not primary and "primary_instrument" in llm_plan:
        # Extract instrument name from strings like "Solo Oboe, espressivo"
        raw = llm_plan["primary_instrument"]
        primary = _normalize_instrument_name(raw)
    if primary:
        _add_role(primary, "melody", None, 1.0)
    # Standard doublings list
    for dbl in melody_data.get("doublings", []):
        _add_role(dbl, "doubling", primary, 0.8)
    # Mahler Conductor format: doubling is a top-level description string
    if "doubling" in llm_plan and isinstance(llm_plan["doubling"], str):
        dbl_instruments = _extract_instrument_names(llm_plan["doubling"])
        for dbl in dbl_instruments:
            _add_role(dbl, "doubling", primary, 0.8)

    # Bass
    bass_data = llm_plan.get("bass", {})
    bass_primary = bass_data.get("primary")
    if bass_primary:
        _add_role(bass_primary, "bass", None, 0.9)
    for dbl in bass_data.get("doublings", []):
        _add_role(dbl, "bass", bass_primary, 0.85)

    # Harmony
    for inst_name in llm_plan.get("harmony", []):
        _add_role(inst_name, "harmony", None, 0.7)

    # Countermelody
    counter = llm_plan.get("countermelody")
    if counter:
        _add_role(counter, "countermelody", None, 0.85)

    # Validate: must have at least melody + bass
    has_melody = any(r.role == "melody" for r in plan.roles)
    has_bass = any(r.role == "bass" for r in plan.roles)
    if not (has_melody and has_bass):
        logger.warning(
            f"LLM plan rejected: missing {'melody' if not has_melody else 'bass'}. "
            f"Plan keys: {list(llm_plan.keys())}"
        )
        return None

    # Store LLM advice (standard + Mahler Conductor format)
    advice = llm_plan.get("advice", "")
    if advice:
        plan.kb_advice.append(f"[llm] {advice[:300]}")
    justification = llm_plan.get("justification", "")
    if justification:
        plan.kb_advice.append(f"[justification] {justification[:300]}")
    harmonic_texture = llm_plan.get("harmonic_texture", "")
    if harmonic_texture:
        plan.kb_advice.append(f"[texture] {harmonic_texture[:300]}")

    return plan
