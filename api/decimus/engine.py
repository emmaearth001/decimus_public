"""Decimus Engine: model loading and full orchestration pipeline."""

import os
import sys
import time

import torch

# Ensure fairseq paths are available
_src_dir = os.path.join(os.path.dirname(__file__), '..')
_fairseq_dir = os.path.join(_src_dir, 'fairseq')
if _fairseq_dir not in sys.path:
    sys.path.insert(0, _fairseq_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from .analyzer import analyze_piano
from .orchestrator import orchestrate_direct
from .planner import create_plan

# Model configuration (matches gen_batch.py)
MAX_POS_LEN = 4096
PI_LEVEL = 2
IGNORE_META_LOSS = 1
BPE = "_bpe"

DATA_BIN = f"linear_{MAX_POS_LEN}_chord{BPE}_hardloss{IGNORE_META_LOSS}"
CHECKPOINT_SUFFIX = f"{DATA_BIN}_PI{PI_LEVEL}"

# Resolve paths relative to the api/ package root, not the caller's cwd
from pathlib import Path as _Path

_PKG_ROOT = _Path(__file__).resolve().parents[1]

DATA_BIN_DIR = str(_PKG_ROOT / "data" / "model_spec" / DATA_BIN / "bin") + "/"
DATA_VOC_DIR = str(_PKG_ROOT / "data" / "model_spec" / DATA_BIN / "vocabs") + "/"
BPE_DIR = str(_PKG_ROOT / "data" / "bpe_res") + "/"
CHECKPOINT_PATH = str(_PKG_ROOT / "ckpt" / f"checkpoint_last_{CHECKPOINT_SUFFIX}.pt")
DEFAULT_OUTPUT_DIR = str(_PKG_ROOT / "generated" / CHECKPOINT_SUFFIX) + "/"


class DecimusEngine:
    """Singleton engine that loads the model once and runs orchestration."""

    _instance = None
    _model = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, project_root: str | None = None):
        """Load the SymphonyNet model and vocabularies."""
        if self._loaded:
            return

        if project_root:
            os.chdir(project_root)

        # Load vocabularies into the shared music_dict
        from gen_utils import music_dict
        music_dict.load_vocabs_bpe(DATA_VOC_DIR, BPE_DIR if BPE == '_bpe' else None)

        # Load model via fairseq
        from fairseq.models import FairseqLanguageModel
        custom_lm = FairseqLanguageModel.from_pretrained(
            '.',
            checkpoint_file=CHECKPOINT_PATH,
            data_name_or_path=DATA_BIN_DIR,
            user_dir="src/fairseq/linear_transformer_inference",
        )

        self._model = custom_lm.models[0]

        # Select device
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        self._model.to(device)
        self._model.eval()
        self._loaded = True
        self._device = device

    @property
    def model(self):
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call engine.load() first.")
        return self._model

    def orchestrate(
        self,
        midi_path: str,
        style: str = "tchaikovsky",
        ensemble: str = "full",
        output_path: str | None = None,
        use_llm: bool = True,
    ) -> dict:
        """Orchestrate a piano MIDI into a full score.

        Directly distributes the original piano notes across orchestral
        instruments, preserving the exact melody, harmony, and bass.

        When use_llm is True, queries the fine-tuned Decimus LLM for
        intelligent instrument assignment before falling back to rules.

        Returns a dict with output path and summary info.
        """
        # Step 1: Analyze piano input
        analysis = analyze_piano(midi_path)

        # Step 2: Create orchestration plan (LLM-assisted or rule-based)
        plan = create_plan(analysis, style_name=style, ensemble_name=ensemble,
                           use_llm=use_llm)

        # Step 3: Build output path
        if output_path is None:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            timestamp = time.strftime("%m-%d_%H-%M-%S", time.localtime())
            midi_name = os.path.splitext(os.path.basename(midi_path))[0]
            output_path = os.path.join(
                DEFAULT_OUTPUT_DIR,
                f"{midi_name}_{style}_{ensemble}_{timestamp}.mid",
            )

        # Step 4: Orchestrate — distribute original notes to instruments
        result = orchestrate_direct(analysis, plan, output_path)

        # Add analysis info to result
        result["style"] = style
        result["ensemble"] = ensemble
        result["analysis"] = {
            "key": analysis.key,
            "tempo": analysis.tempo,
            "time_sig": f"{analysis.time_sig[0]}/{analysis.time_sig[1]}",
            "total_measures": analysis.total_measures,
            "melody_notes": len(analysis.melody_notes),
            "bass_notes": len(analysis.bass_notes),
            "inner_notes": len(analysis.inner_notes),
        }
        result["kb_advice"] = plan.kb_advice
        result["llm_used"] = any("[llm]" in a for a in plan.kb_advice)

        return result
