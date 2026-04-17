"""Decimus API — FastAPI backend for the orchestration pipeline."""

import logging
import os
import sys
import tempfile
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decimus.analyzer import PianoAnalysis, analyze_piano
from decimus.harmonizer import harmonize_melody
from decimus.instruments import ENSEMBLES
from decimus.knowledge import query_rules
from decimus.llm_client import DecimusLLM
from decimus.orchestrator import orchestrate_direct
from decimus.planner import create_plan
from decimus.renderer import is_available as renderer_available
from decimus.renderer import render_midi_to_wav
from decimus.styles import STYLES

# ---------------------------------------------------------------------------
# In-memory log buffer — keeps last 500 entries, queryable via /api/logs
# ---------------------------------------------------------------------------

class LogBuffer(logging.Handler):
    """Captures log records into an in-memory ring buffer."""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.records: deque[dict] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        self.records.append({
            "ts": datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3],
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "exc": self.format_exception(record),
        })

    @staticmethod
    def format_exception(record: logging.LogRecord) -> str | None:
        if record.exc_info and record.exc_info[1]:
            return "".join(traceback.format_exception(*record.exc_info))
        return None


log_buffer = LogBuffer()
log_buffer.setLevel(logging.DEBUG)

# Configure root logger to feed into our buffer
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().addHandler(log_buffer)

# Quiet noisy loggers
for noisy in ("uvicorn.access", "watchfiles", "httpcore", "httpx", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("decimus.web")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Decimus", description="AI Orchestration Engine")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "generated" / "web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOUNDFONT_DIR = Path(__file__).resolve().parents[2] / "data" / "soundfonts"
SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache of last orchestration per session (simple single-user cache)
# Stores: {"analysis": PianoAnalysis, "plan": OrchestrationPlan, "midi_path": str, ...}
_last_orchestration: dict = {}


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(f">>> {request.method} {request.url.path}")
    try:
        response = await call_next(request)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"<<< {request.method} {request.url.path} — CRASHED ({elapsed:.0f}ms): {e}")
        raise
    elapsed = (time.time() - start) * 1000
    level = "info" if response.status_code < 400 else "warning" if response.status_code < 500 else "error"
    getattr(logger, level)(
        f"<<< {request.method} {request.url.path} — {response.status_code} ({elapsed:.0f}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/options")
async def get_options():
    """Return available styles and ensembles."""
    return {
        "styles": [
            {"name": name, "description": style.description}
            for name, style in STYLES.items()
        ],
        "ensembles": [
            {"name": name, "instruments": keys}
            for name, keys in ENSEMBLES.items()
        ],
    }


@app.get("/api/logs")
async def get_logs(level: str = "DEBUG", limit: int = 100):
    """Return recent log entries. Filter by level: DEBUG, INFO, WARNING, ERROR."""
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    min_level = level_order.get(level.upper(), 0)

    entries = [
        r for r in log_buffer.records
        if level_order.get(r["level"], 0) >= min_level
    ]
    return {"logs": list(entries)[-limit:], "total": len(entries)}


@app.get("/api/health")
async def health():
    """Health check with system info."""
    import torch
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "audio_renderer": renderer_available(),
        "output_dir": str(OUTPUT_DIR),
        "generated_files": len(list(OUTPUT_DIR.glob("*.mid"))),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze a MIDI file and return musical content."""
    logger.info(f"Analyzing: {file.filename} ({file.size} bytes)")
    tmp = _save_upload(file)
    try:
        analysis = analyze_piano(tmp)
        logger.info(
            f"Analysis complete: key={analysis.key}, tempo={analysis.tempo:.0f}, "
            f"measures={analysis.total_measures}, "
            f"notes=[mel:{len(analysis.melody_notes)}, bass:{len(analysis.bass_notes)}, "
            f"inner:{len(analysis.inner_notes)}]"
        )
        return _analysis_to_dict(analysis)
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp)


@app.post("/api/orchestrate")
async def orchestrate(
    file: UploadFile = File(...),
    style: str = Form("tchaikovsky"),
    ensemble: str = Form("full"),
    use_llm: bool = Form(False),
):
    """Run full orchestration pipeline on uploaded MIDI."""
    logger.info(f"Orchestrating: {file.filename} | style={style} | ensemble={ensemble} | llm={use_llm}")

    if style not in STYLES:
        logger.warning(f"Unknown style requested: {style}")
        raise HTTPException(400, f"Unknown style: {style}")
    if ensemble not in ENSEMBLES:
        logger.warning(f"Unknown ensemble requested: {ensemble}")
        raise HTTPException(400, f"Unknown ensemble: {ensemble}")

    tmp = _save_upload(file)
    try:
        # Step 1: Analyze
        t0 = time.time()
        analysis = analyze_piano(tmp)
        t_analyze = time.time() - t0
        logger.info(
            f"  [1/3] Analyzed in {t_analyze:.2f}s — key={analysis.key}, "
            f"{len(analysis.melody_notes)} melody, {len(analysis.bass_notes)} bass, "
            f"{len(analysis.inner_notes)} inner"
        )

        # Step 2: Plan
        t0 = time.time()
        plan = create_plan(
            analysis,
            style_name=style,
            ensemble_name=ensemble,
            use_knowledge_base=True,
            use_llm=use_llm,
        )
        t_plan = time.time() - t0
        roles_summary = ", ".join(f"{r.spec.display_name}({r.role})" for r in plan.roles)
        logger.info(f"  [2/3] Planned in {t_plan:.2f}s — {len(plan.roles)} roles: {roles_summary}")
        if plan.kb_advice:
            logger.debug(f"  KB advice: {len(plan.kb_advice)} entries")

        # Step 3: Orchestrate
        t0 = time.time()
        timestamp = time.strftime("%m%d_%H%M%S")
        midi_name = Path(file.filename or "upload").stem
        out_name = f"{midi_name}_{style}_{ensemble}_{timestamp}.mid"
        out_path = str(OUTPUT_DIR / out_name)

        result = orchestrate_direct(analysis, plan, out_path)
        t_orch = time.time() - t0
        logger.info(
            f"  [3/3] Orchestrated in {t_orch:.2f}s — "
            f"{result['total_notes']} notes, {result['num_tracks']} tracks"
        )

        # Step 4: Render audio (if FluidSynth available)
        audio_url = None
        t_render = 0.0
        if renderer_available():
            t0 = time.time()
            audio_name = out_name.replace(".mid", ".wav")
            audio_path = str(OUTPUT_DIR / audio_name)
            try:
                sf_path = _get_active_soundfont()
                render_midi_to_wav(out_path, audio_path, soundfont=sf_path)
                audio_url = f"/api/audio/{audio_name}"
                t_render = time.time() - t0
                logger.info(f"  [4/4] Rendered audio in {t_render:.2f}s")
            except Exception as e:
                logger.warning(f"  Audio rendering skipped: {e}")

        total_time = t_analyze + t_plan + t_orch + t_render
        logger.info(
            f"  Total pipeline: {total_time:.2f}s | Output: {out_name}"
        )

        # Cache for refinement
        _last_orchestration.update({
            "analysis": analysis,
            "plan": plan,
            "style": style,
            "ensemble": ensemble,
            "midi_path": out_path,
            "out_name_base": f"{midi_name}_{style}_{ensemble}",
        })

        return {
            "analysis": _analysis_to_dict(analysis),
            "plan": {
                "style": style,
                "ensemble": ensemble,
                "roles": [
                    {
                        "instrument": r.spec.display_name,
                        "role": r.role,
                        "family": r.spec.family,
                        "velocity_scale": r.velocity_scale,
                        "doubles": r.doubles,
                    }
                    for r in plan.roles
                ],
                "kb_advice": plan.kb_advice,
            },
            "result": {
                "total_notes": result["total_notes"],
                "num_tracks": result["num_tracks"],
                "tracks": result["tracks"],
                "track_notes": result.get("track_notes", {}),
                "download_url": f"/api/download/{out_name}",
                "audio_url": audio_url,
                "tempo": analysis.tempo,
                "ticks_per_beat": analysis.ticks_per_beat,
                "time_sig": [analysis.time_sig[0], analysis.time_sig[1]],
                "total_measures": analysis.total_measures,
            },
        }
    except Exception as e:
        logger.error(f"Orchestration FAILED for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp)


@app.post("/api/reharmonize")
async def reharmonize(
    file: UploadFile = File(...),
    style: str = Form("tchaikovsky"),
):
    """Re-harmonize melody for string orchestra."""
    logger.info(f"Reharmonizing: {file.filename} | style={style}")
    tmp = _save_upload(file)
    try:
        analysis = analyze_piano(tmp)
        timestamp = time.strftime("%m%d_%H%M%S")
        midi_name = Path(file.filename or "upload").stem
        out_name = f"{midi_name}_reharmonized_{timestamp}.mid"
        out_path = str(OUTPUT_DIR / out_name)

        result = harmonize_melody(analysis, style=style, output_path=out_path)
        logger.info(
            f"Reharmonized: {result['total_notes']} notes, "
            f"{result['num_tracks']} tracks, {len(result.get('chords', []))} chords"
        )

        return {
            "analysis": _analysis_to_dict(analysis),
            "result": {
                "total_notes": result["total_notes"],
                "num_tracks": result["num_tracks"],
                "tracks": result["tracks"],
                "chords": result.get("chords", []),
                "download_url": f"/api/download/{out_name}",
            },
        }
    except Exception as e:
        logger.error(f"Reharmonization FAILED for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp)


@app.get("/api/download/{filename}")
async def download(filename: str):
    """Download a generated MIDI file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        logger.warning(f"Download requested for missing file: {filename}")
        raise HTTPException(404, "File not found")
    logger.info(f"Download: {filename}")
    return FileResponse(
        path,
        media_type="audio/midi",
        filename=filename,
    )


@app.get("/api/audio/{filename}")
async def audio(filename: str):
    """Stream a rendered audio file (WAV)."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    media = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
    return FileResponse(path, media_type=media, filename=filename)


@app.post("/api/refine")
async def refine(
    feedback: str = Form(...),
    measure_start: int = Form(0),
    measure_end: int = Form(0),
):
    """Re-orchestrate with user feedback applied to specific measures.

    Parses natural language feedback and adjusts orchestration parameters:
    - "louder" / "quieter" → velocity scaling
    - "remove brass" / "no brass" → filter instruments by family
    - "more strings" / "add strings" → boost string velocity
    - "less busy" / "thinner" → reduce harmony instrument count
    - "more dramatic" → boost all + add percussion emphasis
    """
    if not _last_orchestration:
        raise HTTPException(400, "No previous orchestration to refine. Orchestrate first.")

    analysis = _last_orchestration["analysis"]
    plan = _last_orchestration["plan"]
    style = _last_orchestration["style"]
    ensemble = _last_orchestration["ensemble"]

    logger.info(f"Refining: '{feedback}' | measures {measure_start}-{measure_end}")

    # Parse feedback into orchestration adjustments
    adjustments = _parse_feedback(feedback)
    logger.info(f"  Parsed adjustments: {adjustments}")

    # Determine affected measure range (0 = all measures)
    if measure_end <= measure_start:
        measure_start = 0
        measure_end = analysis.total_measures

    # Apply adjustments to a copy of the plan's roles
    import copy
    refined_plan = copy.deepcopy(plan)
    tpb = analysis.ticks_per_beat
    ticks_per_measure = tpb * analysis.time_sig[0]
    section_start = measure_start * ticks_per_measure
    section_end = measure_end * ticks_per_measure

    _apply_adjustments(refined_plan, adjustments)

    # Re-orchestrate with adjusted plan
    t0 = time.time()
    timestamp = time.strftime("%m%d_%H%M%S")
    base = _last_orchestration["out_name_base"]
    out_name = f"{base}_refined_{timestamp}.mid"
    out_path = str(OUTPUT_DIR / out_name)

    from decimus.orchestrator import orchestrate_direct
    result = orchestrate_direct(analysis, refined_plan, out_path)
    t_orch = time.time() - t0

    # Render audio
    audio_url = None
    if renderer_available():
        t0 = time.time()
        audio_name = out_name.replace(".mid", ".wav")
        audio_path = str(OUTPUT_DIR / audio_name)
        sf_path = _get_active_soundfont()
        try:
            render_midi_to_wav(out_path, audio_path, soundfont=sf_path)
            audio_url = f"/api/audio/{audio_name}"
            logger.info(f"  Refined + rendered in {time.time() - t0 + t_orch:.2f}s")
        except Exception as e:
            logger.warning(f"  Audio rendering skipped: {e}")

    # Update cache
    _last_orchestration["plan"] = refined_plan
    _last_orchestration["midi_path"] = out_path

    return {
        "feedback_applied": adjustments,
        "measures": {"start": measure_start, "end": measure_end},
        "plan": {
            "style": style,
            "ensemble": ensemble,
            "roles": [
                {
                    "instrument": r.spec.display_name,
                    "role": r.role,
                    "family": r.spec.family,
                    "velocity_scale": r.velocity_scale,
                    "doubles": r.doubles,
                }
                for r in refined_plan.roles
            ],
        },
        "result": {
            "total_notes": result["total_notes"],
            "num_tracks": result["num_tracks"],
            "tracks": result["tracks"],
            "track_notes": result.get("track_notes", {}),
            "download_url": f"/api/download/{out_name}",
            "audio_url": audio_url,
            "tempo": analysis.tempo,
            "ticks_per_beat": analysis.ticks_per_beat,
            "time_sig": [analysis.time_sig[0], analysis.time_sig[1]],
            "total_measures": analysis.total_measures,
        },
    }


@app.post("/api/soundfont")
async def upload_soundfont(file: UploadFile = File(...)):
    """Upload a custom SoundFont (.sf2) for audio rendering."""
    if not file.filename or not file.filename.lower().endswith(".sf2"):
        raise HTTPException(400, "Please upload a .sf2 SoundFont file.")

    content = file.file.read()
    max_sf = 500 * 1024 * 1024  # 500 MB
    if len(content) > max_sf:
        raise HTTPException(400, f"SoundFont too large ({len(content)/(1024*1024):.0f} MB). Max 500 MB.")
    if len(content) < 100:
        raise HTTPException(400, "File too small to be a valid SoundFont.")

    # Save to soundfonts directory
    sf_path = SOUNDFONT_DIR / file.filename
    sf_path.write_bytes(content)
    logger.info(f"SoundFont uploaded: {file.filename} ({len(content)/(1024*1024):.1f} MB)")

    # Set as active soundfont
    _last_orchestration["custom_soundfont"] = str(sf_path)

    return {
        "name": file.filename,
        "size_mb": round(len(content) / (1024 * 1024), 1),
        "path": str(sf_path),
    }


@app.get("/api/soundfonts")
async def list_soundfonts():
    """List available SoundFonts."""
    from decimus.renderer import DEFAULT_SF2
    fonts = []
    for sf in sorted(SOUNDFONT_DIR.glob("*.sf2")):
        fonts.append({
            "name": sf.name,
            "size_mb": round(sf.stat().st_size / (1024 * 1024), 1),
            "active": str(sf) == _last_orchestration.get("custom_soundfont", str(DEFAULT_SF2)),
        })
    return {"soundfonts": fonts}


@app.post("/api/soundfont/select")
async def select_soundfont(name: str = Form(...)):
    """Select a previously uploaded SoundFont as active."""
    sf_path = SOUNDFONT_DIR / name
    if not sf_path.exists():
        raise HTTPException(404, f"SoundFont not found: {name}")
    _last_orchestration["custom_soundfont"] = str(sf_path)
    logger.info(f"SoundFont selected: {name}")
    return {"active": name}


# ---------------------------------------------------------------------------
# Symphony AI Chat
# ---------------------------------------------------------------------------

# Lazy-init LLM client
_chat_llm: DecimusLLM | None = None


def _get_chat_llm() -> DecimusLLM:
    global _chat_llm
    if _chat_llm is None:
        _chat_llm = DecimusLLM()
    return _chat_llm


@app.post("/api/chat")
async def chat(question: str = Form(...)):
    """Symphony AI — answer orchestration questions using RAG + LLM."""
    logger.info(f"Chat: '{question[:80]}'")

    # Step 1: Query RAG knowledge base for relevant context
    try:
        advice = query_rules(question, n_results=5)
        rag_context = "\n\n".join(
            f"[{a.source}] {a.text}" for a in advice if a.text.strip()
        )
    except Exception as e:
        logger.warning(f"RAG query failed: {e}")
        rag_context = ""

    # Step 2: Try LLM with RAG context for a synthesized answer
    llm = _get_chat_llm()
    llm_answer = None
    if llm.is_available():
        enriched_question = question
        if rag_context:
            enriched_question = (
                f"Based on the following orchestration knowledge:\n\n"
                f"{rag_context[:2000]}\n\n"
                f"Answer this question: {question}"
            )
        llm_answer = llm.query_advice(enriched_question)

    # Build response
    if llm_answer:
        answer = llm_answer
        source = "llm+rag"
    elif rag_context:
        # Format RAG results as a direct answer
        snippets = []
        for a in advice[:3]:
            text = a.text.strip()
            if len(text) > 300:
                text = text[:300] + "..."
            src = f" ({a.source})" if a.source else ""
            snippets.append(f"{text}{src}")
        answer = "\n\n".join(snippets)
        source = "rag"
    else:
        answer = (
            "I don't have specific information about that. Try asking about "
            "instrument ranges, doublings, orchestral textures, or specific "
            "composers' orchestration techniques."
        )
        source = "fallback"

    logger.info(f"Chat response: source={source}, length={len(answer)}")
    return {"answer": answer, "source": source}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIDI_HEADER = b"MThd"  # All valid MIDI files start with this
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB — covers most piano MIDIs


def _validate_midi(content: bytes, filename: str) -> None:
    """Validate that the uploaded file is actually a MIDI file."""
    if len(content) == 0:
        raise HTTPException(400, "Empty file uploaded. Please upload a MIDI file (.mid).")
    if len(content) > MAX_UPLOAD_BYTES:
        mb = len(content) / (1024 * 1024)
        raise HTTPException(
            400,
            f"File too large ({mb:.1f} MB). Maximum is {MAX_UPLOAD_BYTES // (1024*1024)} MB. "
            f"Try a shorter excerpt or a single-piano MIDI.",
        )
    if len(content) < 14:
        raise HTTPException(
            400,
            f"File too small to be a MIDI file ({len(content)} bytes). "
            f"Please upload a valid .mid file.",
        )
    if not content[:4] == MIDI_HEADER:
        # Try to give a helpful message about what it actually is
        ext = Path(filename).suffix.lower()
        if ext in (".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"):
            detail = (
                f"'{filename}' is an audio file ({ext}), not a MIDI file. "
                f"Decimus needs a .mid/.midi file. You can convert audio to MIDI "
                f"using tools like BasicPitch or piano transcription software."
            )
        elif ext in (".pdf", ".png", ".jpg", ".jpeg", ".gif"):
            detail = f"'{filename}' is a {ext} file, not a MIDI file. Please upload a .mid file."
        elif ext in (".xml", ".musicxml", ".mxl"):
            detail = (
                f"'{filename}' is a MusicXML file. Decimus currently only accepts MIDI. "
                f"Export as MIDI from your notation software (MuseScore, Finale, Sibelius)."
            )
        else:
            detail = (
                f"'{filename}' does not appear to be a valid MIDI file "
                f"(expected MThd header, got {content[:4]!r}). "
                f"Please upload a .mid or .midi file."
            )
        raise HTTPException(400, detail)


def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = Path(file.filename or "upload.mid").suffix or ".mid"
    content = file.file.read()

    # Validate before saving
    _validate_midi(content, file.filename or "upload")
    logger.debug(f"Upload validated: {file.filename} ({len(content)} bytes, header OK)")

    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return tmp


def _get_active_soundfont() -> str | None:
    """Return the currently active SoundFont path, or None for default."""
    custom = _last_orchestration.get("custom_soundfont")
    if custom and os.path.exists(custom):
        return custom
    return None


def _parse_feedback(feedback: str) -> dict:
    """Parse natural language feedback into orchestration adjustments.

    Returns a dict of adjustment keys:
        velocity_scale: float multiplier (e.g. 1.3 for louder)
        remove_families: list of families to mute
        boost_families: list of families to boost
        thin: bool (reduce harmony count)
        dramatic: bool (boost everything)
    """
    fb = feedback.lower().strip()
    adj: dict = {}

    # Volume
    if any(w in fb for w in ("louder", "more volume", "boost", "stronger", "more power")):
        adj["velocity_scale"] = 1.3
    if any(w in fb for w in ("quieter", "softer", "less volume", "gentle", "delicate")):
        adj["velocity_scale"] = 0.65

    # Family removal
    remove = []
    if any(w in fb for w in ("remove brass", "no brass", "without brass", "less brass", "cut brass")):
        remove.append("brass")
    if any(w in fb for w in ("remove strings", "no strings", "without strings", "cut strings")):
        remove.append("strings")
    if any(w in fb for w in ("remove woodwinds", "no woodwinds", "remove winds", "no winds", "cut winds")):
        remove.append("woodwinds")
    if any(w in fb for w in ("remove percussion", "no percussion", "no drums", "remove drums", "cut drums")):
        remove.append("percussion")
    if remove:
        adj["remove_families"] = remove

    # Family boosting
    boost = []
    if any(w in fb for w in ("more strings", "add strings", "boost strings", "strings louder")):
        boost.append("strings")
    if any(w in fb for w in ("more brass", "add brass", "boost brass", "brass louder")):
        boost.append("brass")
    if any(w in fb for w in ("more woodwinds", "add woodwinds", "more winds", "winds louder")):
        boost.append("woodwinds")
    if boost:
        adj["boost_families"] = boost

    # Texture
    if any(w in fb for w in ("less busy", "thinner", "simpler", "cleaner", "sparse", "minimal")):
        adj["thin"] = True
    if any(w in fb for w in ("more dramatic", "bigger", "epic", "climactic", "powerful", "full")):
        adj["dramatic"] = True
    if any(w in fb for w in ("more legato", "smoother", "connected")):
        adj["legato"] = True
    if any(w in fb for w in ("more staccato", "shorter", "crisp", "punchy")):
        adj["staccato"] = True

    # If nothing matched, store the raw feedback
    if not adj:
        adj["raw"] = feedback

    return adj


def _apply_adjustments(plan, adjustments: dict) -> None:
    """Modify an OrchestrationPlan in-place based on parsed adjustments."""

    # Velocity scaling
    vel_scale = adjustments.get("velocity_scale", 1.0)
    if vel_scale != 1.0:
        for role in plan.roles:
            role.velocity_scale = min(1.0, max(0.1, role.velocity_scale * vel_scale))

    # Remove instrument families
    for family in adjustments.get("remove_families", []):
        plan.roles = [r for r in plan.roles if r.spec.family != family]

    # Boost families
    for family in adjustments.get("boost_families", []):
        for role in plan.roles:
            if role.spec.family == family:
                role.velocity_scale = min(1.0, role.velocity_scale * 1.25)

    # Thin texture: keep only melody + bass + 2 best harmony
    if adjustments.get("thin"):
        essential = [r for r in plan.roles if r.role in ("melody", "bass", "doubling")]
        harmony = [r for r in plan.roles if r.role == "harmony"]
        counter = [r for r in plan.roles if r.role == "countermelody"]
        plan.roles = essential + counter[:1] + harmony[:2]

    # Dramatic: boost everything + ensure melody doubling
    if adjustments.get("dramatic"):
        for role in plan.roles:
            role.velocity_scale = min(1.0, role.velocity_scale * 1.2)


def _analysis_to_dict(a: PianoAnalysis) -> dict:
    """Convert PianoAnalysis to JSON-serializable dict."""
    return {
        "key": a.key,
        "tempo": round(a.tempo, 1),
        "time_sig": f"{a.time_sig[0]}/{a.time_sig[1]}",
        "total_measures": a.total_measures,
        "ticks_per_beat": a.ticks_per_beat,
        "melody_notes": [
            {"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
            for n in a.melody_notes
        ],
        "bass_notes": [
            {"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
            for n in a.bass_notes
        ],
        "inner_notes": [
            {"pitch": n.pitch, "start": n.start, "end": n.end, "velocity": n.velocity}
            for n in a.inner_notes
        ],
        "chords": [
            {"root": c.root, "quality": c.quality, "measure": c.measure, "label": c.label}
            for c in a.chords
        ],
        "phrase_boundaries": a.phrase_boundaries,
        "note_counts": {
            "melody": len(a.melody_notes),
            "bass": len(a.bass_notes),
            "inner": len(a.inner_notes),
            "total": len(a.melody_notes) + len(a.bass_notes) + len(a.inner_notes),
        },
    }
