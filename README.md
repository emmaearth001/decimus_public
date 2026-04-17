# Decimus

**AI orchestration engine. Piano MIDI in, full symphonic score out.**

Upload a piano sketch, pick a composer style, and Decimus produces a multi-track
orchestral MIDI plus rendered audio — distributing the melody, bass, and inner
voices across strings, woodwinds, brass, and percussion with voice-leading,
countermelody, and timpani that follow harmonic function.

Built as a hybrid system: a fine-tuned LLM handles stylistic judgment
(which instrument doubles the melody in Tchaikovsky), while symbolic rules
enforce hard physical constraints (instrument ranges, monophonicity,
voice-leading cost).

---

## Quickstart

**Requirements**

- Python 3.10–3.12
- Node.js 20+
- [FluidSynth](https://www.fluidsynth.org/) (`brew install fluidsynth` on macOS; `apt install fluidsynth` on Debian/Ubuntu)

**Setup**

```bash
git clone https://github.com/<you>/decimus.git
cd decimus

# Backend
python -m venv venv
source venv/bin/activate
pip install -e "api[test,dev]"

# Frontend
cd frontend && npm install && cd ..
```

Download a SoundFont into `api/data/soundfonts/` (see
[SoundFonts](#soundfonts) below).

**Run**

```bash
./dev.sh
```

Then open <http://localhost:3000>. A sample MIDI is provided at
`api/data/samples/demo.mid`.

Or start the two services manually:

```bash
# Terminal 1 — API on :8000
cd api && python run_web.py

# Terminal 2 — Frontend on :3000
cd frontend && npm run dev
```

---

## Architecture

```
Next.js (:3000)  ──HTTP──>  FastAPI (:8000)
   page.tsx                  decimus.web.app
   └─ lib/api.ts             └─ 4-stage pipeline:
                                1. analyzer   → voice separation, key, phrases
                                2. planner    → LLM + rules → InstrumentRoles
                                3. orchestrator → voice-leading, timpani, dynamics
                                4. renderer   → FluidSynth MIDI → WAV

        LLM (RunPod vLLM, optional)  ←── llm_client
        ChromaDB RAG (optional)      ←── knowledge
```

Each stage passes typed dataclasses (`PianoAnalysis`, `OrchestrationPlan`,
`InstrumentRole`). Stages are independently testable and fall back
gracefully when optional components are unavailable.

### Pipeline

1. **Analyzer** — music21 chordifies the piece, detects a **melody floor** via
   pitch-gap analysis in the upper register, separates soprano / bass / inner
   voices, detects key (Krumhansl-Schmuckler), and finds phrase boundaries.
2. **Planner** — given a `StylePreset`, tries the fine-tuned LLM first and
   falls back to rule-based register matching. Validates LLM output
   (instrument names, doublings) and enforces the ensemble allow-list.
3. **Orchestrator** — distributes notes: melody to melody instruments, bass to
   bass instruments, countermelody as contrary motion to the melody, timpani
   on tonic/dominant on beat 1 with rolls on phrase boundaries, harmony
   distributed by **voice-leading cost minimization**. Adds expression CCs
   (CC1/CC11/CC64), articulations (staccato/legato/accents), and humanizes
   timing and velocity.
4. **Renderer** — shells out to FluidSynth with a SoundFont, produces WAV.

---

## Styles and ensembles

Ten composer styles, each with its own instrument priorities, doubling rules,
percussion behavior, and LLM prompt:

Mozart, Beethoven, Tchaikovsky, Brahms, Mahler, Debussy, Ravel, Stravinsky,
John Williams, Hans Zimmer.

Four ensembles: full orchestra, strings only, chamber, winds.

---

## Natural-language refinement

After an initial orchestration, the refinement chat accepts prompts like:

- `more dramatic` — boosts velocity, enables doublings, adds percussion accents
- `reduce brass` — drops the brass family
- `thin the texture` — removes inner harmony layers
- `make it quieter` — scales velocities down and switches to sparse percussion

Refinement re-runs orchestration from the cached analysis so it's fast.

---

## Optional components

Decimus is designed to work out-of-the-box on the rule-based path. Three
components are optional; the system degrades cleanly when they're missing.

### SoundFonts

Place any General MIDI SoundFont (`.sf2`) in `api/data/soundfonts/`. The
renderer auto-detects what's available. For a strong orchestral default, try:

- [FluidR3_GM](https://member.keymusician.com/Member/FluidR3_GM/index.html)
  — compact, general purpose (~140 MB)
- [Sonatina Symphonic Orchestra](https://github.com/mrbumpy409/Sonatina-Symphonic-Orchestra)
  — better orchestral quality (~500 MB), preferred if present

Without a SoundFont, orchestration still runs and returns MIDI — only the
rendered audio is skipped.

### LLM planner (RunPod vLLM)

To enable stylistically informed instrument assignment, set:

```bash
DECIMUS_LLM_URL=https://api.runpod.ai/v2/<endpoint_id>/openai/v1
DECIMUS_LLM_KEY=<your_runpod_key>
```

Deployment config and handler live in `deploy/`. Without these,
the planner uses deterministic heuristics.

### Knowledge base (ChromaDB RAG)

The `/api/chat` endpoint and planner advice features use a ChromaDB index
built from orchestration textbook excerpts. To enable:

```bash
cd api
python scripts/rebuild_knowledge_base.py
```

This requires source PDFs under `api/data/knowledge_base/pdf/` (not bundled —
supply your own). Without this, `/api/chat` returns empty advice.

---

## Development

```bash
# Backend
cd api
pytest -q                 # 267 tests
ruff check decimus tests  # lint

# Frontend
cd frontend
npx tsc --noEmit          # type check
npm run build             # production build
```

CI runs all of the above on push and PR
(`.github/workflows/ci.yml`).

### Layout

```
decimus/
├── api/                # FastAPI backend (fully self-contained Python)
│   ├── decimus/        # core modules + web/
│   ├── tests/
│   ├── scripts/
│   ├── data/           # samples, knowledge base, soundfonts
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/           # Next.js app
│   └── Dockerfile
├── deploy/             # RunPod serverless handler for the fine-tuned LLM
├── dev.sh              # run both services locally
└── .github/workflows/  # CI
```

---

## Docker

```bash
# Backend
docker build -t decimus-api api
docker run --rm -p 8000:8000 decimus-api

# Frontend
docker build -t decimus-web frontend
docker run --rm -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://localhost:8000 decimus-web
```

---

## License

MIT — see [LICENSE](LICENSE).

This project builds on [SymphonyNet](https://github.com/symphonynet/SymphonyNet)
for multi-track MIDI modeling and [music21](https://web.mit.edu/music21/) for
music analysis.
