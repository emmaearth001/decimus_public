"""Tests for styles.py — orchestration style presets."""

import pytest

from decimus.instruments import INSTRUMENTS
from decimus.styles import STYLES, StylePreset, get_style


class TestGetStyle:
    def test_all_styles_loadable(self):
        for name in ["mozart", "beethoven", "tchaikovsky", "brahms", "mahler",
                      "debussy", "ravel", "stravinsky", "williams", "zimmer"]:
            style = get_style(name)
            assert isinstance(style, StylePreset)
            assert style.name == name

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown style"):
            get_style("jazz")

    def test_style_count(self):
        assert len(STYLES) >= 5


class TestStylePresets:
    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_melody_instruments_valid(self, style_name):
        style = get_style(style_name)
        for inst in style.melody_instruments:
            assert inst in INSTRUMENTS, f"{inst} not in INSTRUMENTS for {style_name}"

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_harmony_instruments_valid(self, style_name):
        style = get_style(style_name)
        for inst in style.harmony_instruments:
            assert inst in INSTRUMENTS, f"{inst} not in INSTRUMENTS for {style_name}"

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_bass_instruments_valid(self, style_name):
        style = get_style(style_name)
        for inst in style.bass_instruments:
            assert inst in INSTRUMENTS, f"{inst} not in INSTRUMENTS for {style_name}"

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_countermelody_instruments_valid(self, style_name):
        style = get_style(style_name)
        for inst in style.countermelody_instruments:
            assert inst in INSTRUMENTS, f"{inst} not in INSTRUMENTS for {style_name}"

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_has_melody_instruments(self, style_name):
        style = get_style(style_name)
        assert len(style.melody_instruments) > 0

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_has_bass_instruments(self, style_name):
        style = get_style(style_name)
        assert len(style.bass_instruments) > 0

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_voice_count_reasonable(self, style_name):
        style = get_style(style_name)
        assert 5 <= style.max_simultaneous_voices <= 25

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_density_positive(self, style_name):
        style = get_style(style_name)
        assert style.density_target > 0

    @pytest.mark.parametrize("style_name", list(STYLES.keys()))
    def test_temperatures_valid(self, style_name):
        style = get_style(style_name)
        assert 0.1 <= style.event_temperature <= 2.0
        assert 0.1 <= style.instrument_temperature <= 2.0
