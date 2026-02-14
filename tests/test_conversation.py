"""Tests for app.conversation module — state management and troubleshooting logic."""
import pytest
from app.conversation import (
    _get_initial_state,
    _serialize_state,
    _deserialize_state,
    infer_appliance_type,
    get_troubleshooting_steps_summary,
    get_next_troubleshooting_prompt,
    is_positive_response,
    BASIC_TROUBLESHOOTING,
)
from datetime import datetime


# ── Initial State ──────────────────────────────────────────────────────

class TestInitialState:
    def test_initial_state_has_required_keys(self):
        state = _get_initial_state()
        required = [
            "step", "appliance_type", "symptoms", "symptom_summary",
            "error_codes", "is_urgent", "troubleshooting_step", "resolved",
            "zip_code", "time_preference", "offered_slots", "customer_phone",
            "appointment_booked", "appointment_id", "no_input_attempts",
            "customer_email", "pending_email", "email_attempts",
            "email_confirm_attempts", "image_upload_sent", "upload_token",
            "waiting_for_upload", "upload_poll_count",
            "understand_attempts", "appliance_attempts", "zip_attempts",
            "customer_name", "troubleshooting_steps_text", "analysis_spoken",
            "upload_wait_attempts",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_initial_step_is_greet_ask_name(self):
        state = _get_initial_state()
        assert state["step"] == "greet_ask_name"

    def test_initial_state_counters_are_zero(self):
        state = _get_initial_state()
        assert state["no_input_attempts"] == 0
        assert state["email_attempts"] == 0
        assert state["understand_attempts"] == 0
        assert state["zip_attempts"] == 0


# ── Serialization / Deserialization ────────────────────────────────────

class TestSerialization:
    def test_serialize_datetime(self):
        state = {"start_time": datetime(2025, 6, 15, 9, 0)}
        result = _serialize_state(state)
        assert isinstance(result["start_time"], str)
        assert "2025-06-15" in result["start_time"]

    def test_serialize_offered_slots(self):
        state = {
            "offered_slots": [
                {"slot_id": 1, "start_time": datetime(2025, 6, 15, 9, 0), "end_time": datetime(2025, 6, 15, 12, 0)}
            ]
        }
        result = _serialize_state(state)
        assert isinstance(result["offered_slots"][0]["start_time"], str)

    def test_deserialize_offered_slots(self):
        state = {
            "offered_slots": [
                {"slot_id": 1, "start_time": "2025-06-15T09:00:00", "end_time": "2025-06-15T12:00:00"}
            ]
        }
        result = _deserialize_state(state)
        assert isinstance(result["offered_slots"][0]["start_time"], datetime)

    def test_roundtrip(self):
        original = _get_initial_state()
        serialized = _serialize_state(original)
        deserialized = _deserialize_state(serialized)
        assert deserialized["step"] == original["step"]
        assert deserialized["no_input_attempts"] == original["no_input_attempts"]


# ── Appliance Inference ────────────────────────────────────────────────

class TestInferApplianceType:
    @pytest.mark.parametrize("text,expected", [
        ("my washer is broken", "washer"),
        ("the washing machine won't start", "washer"),
        ("dryer is making noise", "dryer"),
        ("fridge is not cooling", "refrigerator"),
        ("refrigerator leaking water", "refrigerator"),
        ("dishwasher won't drain", "dishwasher"),
        ("oven not heating up", "oven"),
        ("stove burner broken", "oven"),
        ("ac is not working", "hvac"),
        ("air conditioner blowing warm", "hvac"),
        ("hvac system broken", "hvac"),
    ])
    def test_known_appliances(self, text, expected):
        assert infer_appliance_type(text) == expected

    def test_unknown_input(self):
        assert infer_appliance_type("hello there") is None

    def test_empty_input(self):
        assert infer_appliance_type("") is None


# ── Troubleshooting Steps Summary ──────────────────────────────────────

class TestTroubleshootingStepsSummary:
    def test_known_appliance_returns_steps(self):
        summary = get_troubleshooting_steps_summary("washer")
        assert "Step 1:" in summary
        assert "Step 2:" in summary
        assert "Step 3:" in summary

    def test_unknown_appliance_returns_empty(self):
        assert get_troubleshooting_steps_summary("toaster") == ""

    def test_empty_appliance_returns_empty(self):
        assert get_troubleshooting_steps_summary("") == ""

    def test_none_appliance_returns_empty(self):
        assert get_troubleshooting_steps_summary(None) == ""

    def test_all_appliances_have_steps(self):
        for appliance in BASIC_TROUBLESHOOTING:
            summary = get_troubleshooting_steps_summary(appliance)
            assert len(summary) > 0, f"No summary for {appliance}"

    def test_please_prefix_removed(self):
        summary = get_troubleshooting_steps_summary("washer")
        assert "Please check" not in summary
        assert "Check" in summary


# ── get_next_troubleshooting_prompt ────────────────────────────────────

class TestGetNextTroubleshootingPrompt:
    def test_returns_first_step(self):
        state = {"appliance_type": "washer", "troubleshooting_step": 0}
        prompt = get_next_troubleshooting_prompt(state)
        assert prompt is not None
        assert state["troubleshooting_step"] == 1

    def test_returns_none_after_all_steps(self):
        state = {"appliance_type": "washer", "troubleshooting_step": 100}
        prompt = get_next_troubleshooting_prompt(state)
        assert prompt is None

    def test_unknown_appliance_returns_none(self):
        state = {"appliance_type": "toaster", "troubleshooting_step": 0}
        assert get_next_troubleshooting_prompt(state) is None


# ── Positive Response Detection ────────────────────────────────────────

class TestIsPositiveResponse:
    @pytest.mark.parametrize("text", [
        "yes", "yeah", "yep", "ok", "sure", "absolutely",
        "it worked", "that worked", "fixed", "working now",
        "that helped", "all good", "problem solved",
    ])
    def test_positive_responses(self, text):
        assert is_positive_response(text) is True

    @pytest.mark.parametrize("text", [
        "no", "nope", "not working", "didn't work", "still broken",
        "same problem", "no change", "negative", "unfortunately",
        "didn't help", "doesn't work", "still not working",
    ])
    def test_negative_responses(self, text):
        assert is_positive_response(text) is False

    def test_ambiguous_good_with_negation(self):
        # "The dial is good, it's at max cooling" — issue persists
        assert is_positive_response("not working but the dial is good") is False

    def test_empty_string(self):
        assert is_positive_response("") is False
