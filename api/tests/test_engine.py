"""Tests for engine.py — singleton pattern, pipeline orchestration."""

import pytest

from decimus.engine import DecimusEngine


class TestSingleton:
    def test_same_instance(self):
        # Reset singleton for test isolation
        DecimusEngine._instance = None
        a = DecimusEngine()
        b = DecimusEngine()
        assert a is b

    def test_model_not_loaded_initially(self):
        DecimusEngine._instance = None
        engine = DecimusEngine()
        engine._loaded = False
        with pytest.raises(RuntimeError, match="Engine not loaded"):
            _ = engine.model


class TestOrchestrateMethod:
    def test_orchestrate_produces_result(self, sample_analysis, tmp_midi_path):
        """Test the orchestrate method with a real MIDI file."""
        import os
        midi_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples', 'demo.mid')
        if not os.path.exists(midi_path):
            pytest.skip("demo.mid not available")

        DecimusEngine._instance = None
        engine = DecimusEngine()
        result = engine.orchestrate(
            midi_path, style="tchaikovsky", ensemble="strings",
            output_path=tmp_midi_path, use_llm=False,
        )
        assert "output_path" in result
        assert "style" in result
        assert result["style"] == "tchaikovsky"
        assert "analysis" in result
        assert "key" in result["analysis"]
        assert os.path.exists(tmp_midi_path)
