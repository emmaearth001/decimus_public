"""Decimus LLM client for the fine-tuned orchestration model.

Calls the RunPod serverless vLLM endpoint. Falls back gracefully
if the endpoint is unavailable.

Environment variables:
    DECIMUS_LLM_URL: RunPod endpoint base URL
        (e.g., https://api.runpod.ai/v2/<endpoint_id>/openai/v1)
    DECIMUS_LLM_KEY: RunPod API key
    DECIMUS_LLM_ENABLED: "true" (default) or "false" to disable
"""

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Decimus LLM, an expert orchestration advisor trained on "
    "Rimsky-Korsakov's Principles of Orchestration. You help composers "
    "transform piano sketches into full orchestral scores by recommending "
    "instrument assignments, doublings, voicings, and textures. "
    "When asked for an orchestration plan, respond with structured JSON."
)

# Style-specific system prompts for each composer
# Each prompt must instruct the LLM to return JSON with:
#   melody: {primary, doublings[]}, bass: {primary, doublings[]},
#   harmony: [instruments], countermelody: instrument or null, advice: str
_JSON_FORMAT = (
    "\n\n### Output Format:\n"
    "Respond with a JSON object: "
    '{"melody": {"primary": "instrument", "doublings": []}, '
    '"bass": {"primary": "instrument", "doublings": []}, '
    '"harmony": ["instrument", ...], '
    '"countermelody": "instrument" or null, '
    '"advice": "brief reasoning"}'
)

STYLE_PROMPTS = {
    "mozart": (
        "You are Decimus-Conductor orchestrating in the style of Wolfgang Amadeus Mozart. "
        "Prioritize transparency, elegance, and balance. Strings carry primary material. "
        "Woodwinds are soloistic — oboe and flute for color, not doubling. Brass (horns only) "
        "provides harmonic support sparingly. Avoid thick textures. Favor antiphonal exchanges "
        "between strings and winds. Keep inner voices moving but never muddy. "
        "Classical-era restraint: fewer instruments, each with a clear purpose." + _JSON_FORMAT
    ),
    "beethoven": (
        "You are Decimus-Conductor orchestrating in the style of Ludwig van Beethoven. "
        "Balance Classical clarity with Romantic power. Strings are the backbone. "
        "Use horns for heroic weight and timpani for rhythmic drive. Build dramatic "
        "crescendos by adding instruments layer by layer. Sforzando accents are key. "
        "Cellos and basses often double for power. Woodwinds can carry lyrical second themes. "
        "Contrast extreme dynamics — whisper to thunder." + _JSON_FORMAT
    ),
    "tchaikovsky": (
        "You are Decimus-Conductor orchestrating in the style of Pyotr Ilyich Tchaikovsky. "
        "Rich, emotional, sweeping sonorities. Violin I carries soaring melodies, often "
        "doubled by flute an octave higher for brilliance. Cellos sing countermelodies. "
        "Full horn section for warmth. Brass for climactic peaks. Use harp for color. "
        "Woodwinds in thirds for pastoral moments. Build to massive tutti with cymbal crashes. "
        "Emotional directness over subtlety." + _JSON_FORMAT
    ),
    "brahms": (
        "You are Decimus-Conductor orchestrating in the style of Johannes Brahms. "
        "Warm, thick, autumnal textures. Favor the middle register — violas and clarinets "
        "are central. Horns are essential for Brahms's harmonic cushion. Avoid shrill high "
        "registers. Double at the unison for warmth, not the octave. Cross-rhythm between "
        "voices. Cellos often carry the melody. Restrained brass — no bombast, only weight. "
        "Dense but never muddy inner voices." + _JSON_FORMAT
    ),
    "mahler": (
        "You are Decimus-Conductor orchestrating in the style of Gustav Mahler. "
        "Polyphonic transparency: every voice has a distinct color. If two instruments "
        "play the same note, there must be a timbral reason (oboe + muted trumpet for "
        "'piercing' effect). Extreme dynamics with surgical precision — con sordino horns "
        "for ppp, solo bassoon in high register for eerie quiet. 'Wunderhorn' texture: "
        "solo woodwinds (oboe, English horn, E-flat clarinet) and nature calls. "
        "Register hierarchy: high melody = sparse/low accompaniment for 'hollow' space." + _JSON_FORMAT
    ),
    "debussy": (
        "You are Decimus-Conductor orchestrating in the style of Claude Debussy. "
        "Impressionistic color over melody. Flute is the primary voice — airy, floating. "
        "Harp and strings provide shimmering backgrounds (tremolo, harmonics, divisi). "
        "No heavy brass. Clarinets in chalumeau register for warmth. Avoid strong downbeats. "
        "Woodwinds in parallel motion (thirds, fourths). Muted strings for veiled textures. "
        "Oboe for brief solos. Everything should sound like light filtering through water." + _JSON_FORMAT
    ),
    "ravel": (
        "You are Decimus-Conductor orchestrating in the style of Maurice Ravel. "
        "Master colorist — every instrument chosen for its unique timbre. Precise, "
        "jewel-like orchestration. Solo instruments featured prominently. Celesta and harp "
        "for sparkle. Woodwinds in their most characteristic registers. Muted brass for "
        "distant color. String harmonics and pizzicato for texture. Build Bolero-like "
        "crescendos by adding timbres progressively. Clean, transparent, never thick." + _JSON_FORMAT
    ),
    "stravinsky": (
        "You are Decimus-Conductor orchestrating in the style of Igor Stravinsky. "
        "Bold, angular, percussive. Exploit extreme registers — bassoon in its highest "
        "range, piccolo in its lowest. Favor dry, pointillistic textures. Heavy rhythmic "
        "accents from brass and percussion. No romantic blending — each instrument "
        "stands apart. Ostinatos in low brass. Woodwinds in unusual combinations "
        "(clarinet + trumpet). Strings as percussion (col legno, pizzicato). "
        "Asymmetric rhythms and sharp dynamic contrasts." + _JSON_FORMAT
    ),
    "williams": (
        "You are Decimus-Conductor orchestrating in the style of John Williams. "
        "Cinematic, heroic, emotionally direct. Soaring violin melodies doubled by "
        "French horns for power. Full brass fanfares with trumpets leading. Lush string "
        "divisi for tender moments. Harp arpeggios for magic/wonder. Timpani and cymbals "
        "for dramatic punctuation. Woodwinds for whimsy and lightness. Layer instruments "
        "for massive crescendos. Everything serves the emotional narrative." + _JSON_FORMAT
    ),
    "zimmer": (
        "You are Decimus-Conductor orchestrating in the style of Hans Zimmer. "
        "Modern cinematic power. Heavy low strings (cellos + basses in unison) for "
        "driving ostinatos. Brass stabs and swells for impact. Minimal woodwinds — "
        "this is about weight, not color. Rhythmic percussion patterns (taiko-style). "
        "Simple harmonic language, massive sound. Violin I for soaring emotional peaks. "
        "Layer upon layer of the same idea for overwhelming force. "
        "Drones and pedal points for tension." + _JSON_FORMAT
    ),
}


def get_system_prompt(style: str = "") -> str:
    """Return the appropriate system prompt for the given style."""
    return STYLE_PROMPTS.get(style, SYSTEM_PROMPT)


def _extract_base_url(url: str) -> str:
    """Extract the RunPod base URL (without /openai/v1 suffix)."""
    url = url.rstrip("/")
    if url.endswith("/openai/v1"):
        return url[:-len("/openai/v1")]
    return url


@dataclass
class LLMConfig:
    api_url: str = ""
    api_key: str = ""
    model_name: str = "emmaearth001/decimus-llm-v1"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 90.0  # RunPod serverless can take time on cold start
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            api_url=os.environ.get("DECIMUS_LLM_URL", ""),
            api_key=os.environ.get("DECIMUS_LLM_KEY", ""),
            enabled=os.environ.get("DECIMUS_LLM_ENABLED", "true").lower() == "true",
        )


class DecimusLLM:
    """Client for the fine-tuned Decimus LLM on RunPod."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if the LLM endpoint is reachable via RunPod health check."""
        if not self.config.enabled or not self.config.api_url:
            return False
        if self._available is not None:
            return self._available

        try:
            import httpx
            base_url = _extract_base_url(self.config.api_url)
            r = httpx.get(
                f"{base_url}/health",
                headers=self._headers(),
                timeout=10.0,
            )
            if r.status_code == 200:
                data = r.json()
                workers = data.get("workers", {})
                self._available = workers.get("ready", 0) > 0 or workers.get("idle", 0) > 0
            else:
                self._available = False
        except Exception:
            self._available = False

        return self._available

    def _run_sync(self, payload: dict) -> dict | None:
        """Submit a job to RunPod and wait for the result (runsync)."""
        import httpx
        base_url = _extract_base_url(self.config.api_url)

        try:
            response = httpx.post(
                f"{base_url}/runsync",
                headers=self._headers(),
                json={"input": payload},
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            if status == "COMPLETED":
                return result.get("output", {})

            # If IN_QUEUE or IN_PROGRESS, poll for result
            job_id = result.get("id")
            if job_id and status in ("IN_QUEUE", "IN_PROGRESS"):
                return self._poll_result(base_url, job_id)

            logger.warning(f"RunPod job failed with status: {status}")
            return None

        except Exception as e:
            logger.warning(f"RunPod request failed: {e}")
            return None

    def _poll_result(self, base_url: str, job_id: str,
                     max_wait: float = 120.0) -> dict | None:
        """Poll RunPod for job completion."""
        import httpx
        start = time.time()
        while time.time() - start < max_wait:
            try:
                r = httpx.get(
                    f"{base_url}/status/{job_id}",
                    headers=self._headers(),
                    timeout=10.0,
                )
                data = r.json()
                if data.get("status") == "COMPLETED":
                    return data.get("output", {})
                if data.get("status") == "FAILED":
                    logger.warning(f"RunPod job failed: {data}")
                    return None
            except Exception:
                pass
            time.sleep(2)
        logger.warning("RunPod job timed out")
        return None

    def query_plan(self, analysis_text: str, style: str = "") -> dict | None:
        """Query the LLM for an orchestration plan.

        Args:
            analysis_text: Formatted analysis string.
            style: Style name for style-specific system prompt.

        Returns parsed JSON dict with plan, or None if unavailable/failed.
        """
        if not self.is_available():
            return None

        sys_prompt = get_system_prompt(style)

        # Try OpenAI-compatible endpoint first
        result = self._try_openai_api(analysis_text, self.config.temperature,
                                       self.config.max_tokens,
                                       system_prompt=sys_prompt)
        if result is not None:
            return self._parse_plan_json(result)

        # Fallback to RunPod native API
        payload = {
            "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{analysis_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        output = self._run_sync(payload)
        if output and "text" in output:
            return self._parse_plan_json(output["text"])
        # vLLM worker may return list of outputs
        if isinstance(output, list) and output:
            text = output[0].get("text", "") if isinstance(output[0], dict) else str(output[0])
            return self._parse_plan_json(text)
        return None

    def query_advice(self, question: str) -> str | None:
        """Query the LLM for free-form orchestration advice."""
        if not self.is_available():
            return None

        result = self._try_openai_api(question, 0.5, 512)
        if result is not None:
            return result

        payload = {
            "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "temperature": 0.5,
            "max_tokens": 512,
        }
        output = self._run_sync(payload)
        if output and "text" in output:
            return output["text"]
        return None

    def _try_openai_api(self, user_content: str, temperature: float,
                         max_tokens: int,
                         system_prompt: str = "") -> str | None:
        """Try the OpenAI-compatible chat completions endpoint."""
        try:
            import httpx
            url = self.config.api_url.rstrip("/")
            if not url.endswith("/openai/v1"):
                url = f"{url}/openai/v1"
            sys_msg = system_prompt or SYSTEM_PROMPT
            response = httpx.post(
                f"{url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.config.model_name,
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _parse_plan_json(text: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown fences."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return None


def format_analysis_for_llm(analysis, style_name: str, ensemble_name: str) -> str:
    """Format a PianoAnalysis into the prompt the LLM expects.

    Must match the format used during training (Source E pairs).
    """
    melody_notes = analysis.melody_notes
    bass_notes = analysis.bass_notes
    inner_notes = analysis.inner_notes

    mel_avg = sum(n.pitch for n in melody_notes) / len(melody_notes) if melody_notes else 60
    bass_avg = sum(n.pitch for n in bass_notes) / len(bass_notes) if bass_notes else 40

    mel_register = "low" if mel_avg < 55 else ("high" if mel_avg > 75 else "mid")
    bass_register = "low" if bass_avg < 55 else ("high" if bass_avg > 75 else "mid")
    density = (
        "sparse" if len(inner_notes) < 20
        else "dense" if len(inner_notes) > 100
        else "moderate"
    )

    chord_str = " - ".join(
        c.label for c in analysis.chords[:16]
        if hasattr(c, 'label') and c.label != "NA"
    )

    return (
        f"Create an orchestration plan for this piano piece.\n\n"
        f"Key: {analysis.key}\n"
        f"Tempo: {analysis.tempo:.0f} BPM\n"
        f"Time signature: {analysis.time_sig[0]}/{analysis.time_sig[1]}\n"
        f"Melody register: {mel_register} (avg pitch {mel_avg:.0f})\n"
        f"Bass register: {bass_register} (avg pitch {bass_avg:.0f})\n"
        f"Inner voice density: {density} ({len(inner_notes)} notes)\n"
        f"Style: {style_name}\n"
        f"Ensemble: {ensemble_name}\n"
        f"Phrase count: {len(analysis.phrase_boundaries)}\n"
        f"Total measures: {analysis.total_measures}\n"
        f"Chords: {chord_str}"
    )
