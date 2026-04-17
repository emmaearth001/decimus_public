"""Tests for llm_client.py — JSON parsing, URL extraction, config, system prompts."""

from decimus.analyzer import PianoAnalysis
from decimus.llm_client import (
    SYSTEM_PROMPT,
    DecimusLLM,
    LLMConfig,
    _extract_base_url,
    format_analysis_for_llm,
    get_system_prompt,
)

# ---------------------------------------------------------------------------
# _extract_base_url
# ---------------------------------------------------------------------------

class TestExtractBaseUrl:
    def test_strips_openai_suffix(self):
        url = "https://api.runpod.ai/v2/abc123/openai/v1"
        assert _extract_base_url(url) == "https://api.runpod.ai/v2/abc123"

    def test_strips_trailing_slash(self):
        url = "https://api.runpod.ai/v2/abc123/openai/v1/"
        assert _extract_base_url(url) == "https://api.runpod.ai/v2/abc123"

    def test_no_suffix_unchanged(self):
        url = "https://api.runpod.ai/v2/abc123"
        assert _extract_base_url(url) == "https://api.runpod.ai/v2/abc123"


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------

class TestGetSystemPrompt:
    def test_default_prompt(self):
        assert get_system_prompt() == SYSTEM_PROMPT

    def test_unknown_style_returns_default(self):
        assert get_system_prompt("jazz") == SYSTEM_PROMPT

    def test_known_styles(self):
        for style in ["mahler", "tchaikovsky", "mozart", "stravinsky", "williams"]:
            prompt = get_system_prompt(style)
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_mahler_prompt_specific(self):
        prompt = get_system_prompt("mahler")
        assert "Mahler" in prompt
        assert "Polyphonic" in prompt


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.api_url == ""
        assert config.enabled is True
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DECIMUS_LLM_URL", "http://test.com")
        monkeypatch.setenv("DECIMUS_LLM_KEY", "test-key")
        monkeypatch.setenv("DECIMUS_LLM_ENABLED", "false")
        config = LLMConfig.from_env()
        assert config.api_url == "http://test.com"
        assert config.api_key == "test-key"
        assert config.enabled is False

    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("DECIMUS_LLM_URL", raising=False)
        monkeypatch.delenv("DECIMUS_LLM_KEY", raising=False)
        monkeypatch.delenv("DECIMUS_LLM_ENABLED", raising=False)
        config = LLMConfig.from_env()
        assert config.api_url == ""
        assert config.enabled is True


# ---------------------------------------------------------------------------
# DecimusLLM._parse_plan_json
# ---------------------------------------------------------------------------

class TestParsePlanJson:
    def test_plain_json(self):
        text = '{"melody": {"primary": "violin_1"}, "bass": {"primary": "cello"}}'
        result = DecimusLLM._parse_plan_json(text)
        assert result is not None
        assert result["melody"]["primary"] == "violin_1"

    def test_json_in_markdown_fence(self):
        text = 'Here is the plan:\n```json\n{"melody": {"primary": "oboe"}}\n```\nDone.'
        result = DecimusLLM._parse_plan_json(text)
        assert result is not None
        assert result["melody"]["primary"] == "oboe"

    def test_json_in_generic_fence(self):
        text = 'Plan:\n```\n{"melody": {"primary": "flute"}}\n```'
        result = DecimusLLM._parse_plan_json(text)
        assert result is not None
        assert result["melody"]["primary"] == "flute"

    def test_invalid_json_returns_none(self):
        text = "This is not valid JSON at all"
        result = DecimusLLM._parse_plan_json(text)
        assert result is None

    def test_empty_string(self):
        result = DecimusLLM._parse_plan_json("")
        assert result is None


# ---------------------------------------------------------------------------
# DecimusLLM.is_available (without network)
# ---------------------------------------------------------------------------

class TestLLMAvailability:
    def test_disabled_config(self):
        config = LLMConfig(enabled=False)
        llm = DecimusLLM(config)
        assert llm.is_available() is False

    def test_no_url(self):
        config = LLMConfig(api_url="", enabled=True)
        llm = DecimusLLM(config)
        assert llm.is_available() is False

    def test_cached_availability(self):
        config = LLMConfig(enabled=False)
        llm = DecimusLLM(config)
        llm.is_available()
        # Second call should use cached value
        assert llm.is_available() is False


# ---------------------------------------------------------------------------
# format_analysis_for_llm
# ---------------------------------------------------------------------------

class TestFormatAnalysisForLLM:
    def test_basic_format(self, sample_analysis):
        text = format_analysis_for_llm(sample_analysis, "tchaikovsky", "full")
        assert "Key: Gm" in text
        assert "tchaikovsky" in text
        assert "full" in text
        assert "Tempo: 85" in text

    def test_empty_analysis(self):
        analysis = PianoAnalysis()
        text = format_analysis_for_llm(analysis, "mozart", "strings")
        assert "Key: C" in text
        assert "mozart" in text

    def test_register_classification(self, sample_analysis):
        text = format_analysis_for_llm(sample_analysis, "tchaikovsky", "full")
        assert "register" in text.lower()
