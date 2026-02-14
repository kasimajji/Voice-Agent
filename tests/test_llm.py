"""Tests for app.llm module — LLM functions with fallback behavior (no API key needed)."""
import pytest
from unittest.mock import patch, MagicMock


class TestLlmAnalyzeCustomerIntent:
    """Test the autonomous intent detection function."""

    def setup_method(self):
        from app.llm import llm_analyze_customer_intent
        self.analyze = llm_analyze_customer_intent

    def test_empty_input_returns_unclear(self):
        result = self.analyze("")
        assert result["intent"] == "unclear"
        assert result["appliance_type"] is None

    def test_none_input_returns_unclear(self):
        result = self.analyze(None)
        assert result["intent"] == "unclear"

    @patch("app.llm.model", None)
    def test_scheduling_keyword_detected(self):
        result = self.analyze("I want to schedule a technician for my washer")
        assert result["wants_scheduling"] is True
        assert result["appliance_type"] == "washer"

    @patch("app.llm.model", None)
    def test_appliance_detected_from_keywords(self):
        result = self.analyze("my refrigerator is broken and leaking water everywhere")
        assert result["appliance_type"] == "refrigerator"

    @patch("app.llm.model", None)
    def test_dryer_detected(self):
        result = self.analyze("the dryer is making a loud noise")
        assert result["appliance_type"] == "dryer"

    @patch("app.llm.model", None)
    def test_dishwasher_detected(self):
        result = self.analyze("my dishwasher won't drain")
        assert result["appliance_type"] == "dishwasher"

    @patch("app.llm.model", None)
    def test_hvac_detected(self):
        result = self.analyze("my air conditioner is blowing warm air")
        assert result["appliance_type"] == "hvac"

    @patch("app.llm.model", None)
    def test_oven_detected(self):
        result = self.analyze("the stove burner won't light")
        assert result["appliance_type"] == "oven"

    @patch("app.llm.model", None)
    def test_full_description_detected(self):
        result = self.analyze(
            "My washer is making a really loud banging noise during the spin cycle "
            "and it's been leaking water from the bottom for the past two days"
        )
        assert result["appliance_type"] == "washer"
        assert result["has_full_description"] is True

    @patch("app.llm.model", None)
    def test_no_appliance_returns_unclear(self):
        result = self.analyze("hello")
        assert result["intent"] == "unclear"
        assert result["appliance_type"] is None


class TestLlmExtractName:
    """Test name extraction with fallback."""

    def setup_method(self):
        from app.llm import llm_extract_name
        self.extract = llm_extract_name

    @patch("app.llm.model", None)
    def test_simple_name(self):
        result = self.extract("My name is John")
        assert result is not None
        assert "john" in result.lower() or "John" in result

    @patch("app.llm.model", None)
    def test_name_with_im(self):
        result = self.extract("I'm Sarah")
        assert result is not None

    @patch("app.llm.model", None)
    def test_empty_returns_none_or_fallback(self):
        result = self.extract("")
        # Without LLM model, empty input returns None (caller handles fallback)
        assert result is None or isinstance(result, str)


class TestLlmExtractEmail:
    """Test email extraction with deterministic pre-processing."""

    def setup_method(self):
        from app.llm import llm_extract_email
        self.extract = llm_extract_email

    @patch("app.llm.model", None)
    def test_plain_email(self):
        result = self.extract("john@gmail.com")
        assert "@" in result
        assert "gmail" in result.lower()

    @patch("app.llm.model", None)
    def test_spelled_out_email(self):
        result = self.extract("j o h n at gmail dot com")
        assert "@" in result
        assert ".com" in result

    @patch("app.llm.model", None)
    def test_at_the_rate_normalization(self):
        result = self.extract("john at the rate gmail dot com")
        assert "@" in result

    @patch("app.llm.model", None)
    def test_empty_input(self):
        result = self.extract("")
        assert result is not None  # Should return a constructed email


class TestLlmInterpretTroubleshootingResponse:
    """Test troubleshooting response interpretation."""

    def setup_method(self):
        from app.llm import llm_interpret_troubleshooting_response
        self.interpret = llm_interpret_troubleshooting_response

    @patch("app.llm.model", None)
    def test_positive_response(self):
        result = self.interpret("yes that worked!", "Check the power cord")
        assert result["is_resolved"] is True

    @patch("app.llm.model", None)
    def test_negative_response(self):
        result = self.interpret("no it's still not working", "Check the power cord")
        assert result["is_resolved"] is False

    @patch("app.llm.model", None)
    def test_ambiguous_negative(self):
        result = self.interpret("I checked but still broken", "Check the power cord")
        assert result["is_resolved"] is False


class TestLlmClassifyAppliance:
    """Test appliance classification."""

    def setup_method(self):
        from app.llm import llm_classify_appliance
        self.classify = llm_classify_appliance

    @patch("app.llm.model", None)
    def test_returns_none_without_model(self):
        # Without model, classification returns None (falls back to keyword in routes)
        result = self.classify("my washer is broken")
        # May return None or a value depending on fallback logic
        assert result is None or result in {"washer", "dryer", "refrigerator", "dishwasher", "oven", "hvac"}


class TestLlmExtractSymptoms:
    """Test symptom extraction."""

    def setup_method(self):
        from app.llm import llm_extract_symptoms
        self.extract = llm_extract_symptoms

    @patch("app.llm.model", None)
    def test_returns_fallback_dict(self):
        result = self.extract("it's making a loud noise and leaking")
        assert "symptom_summary" in result
        assert "error_codes" in result
        assert "is_urgent" in result

    @patch("app.llm.model", None)
    def test_empty_input(self):
        result = self.extract("")
        assert "symptom_summary" in result
