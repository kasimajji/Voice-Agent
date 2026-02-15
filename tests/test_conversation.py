"""Tests for app.conversation module — state management and appliance inference."""
import pytest
from app.conversation import (
    _get_initial_state,
    _serialize_state,
    _deserialize_state,
    infer_appliance_type,
)
from app.llm import llm_interpret_troubleshooting_response
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


# ── Troubleshooting Response Interpretation (keyword fallback) ────────

class TestTroubleshootingResponseInterpretation:
    """Tests the keyword fallback of llm_interpret_troubleshooting_response
    (used when LLM model is None, e.g. in test environments)."""

    @pytest.mark.parametrize("text", [
        "no it didn't work",
        "still not working",
        "nope, same issue",
        "didn't help at all",
        "no change",
        "I checked it but nothing happened",
        "I tried that already",
    ])
    def test_negative_responses_not_resolved(self, text):
        result = llm_interpret_troubleshooting_response(text, "Check the power cord")
        assert result["is_resolved"] is False

    @pytest.mark.parametrize("text", [
        "yes that fixed it",
        "it worked! it's working now",
        "that helped, all good",
        "problem solved",
    ])
    def test_positive_responses_resolved(self, text):
        result = llm_interpret_troubleshooting_response(text, "Check the power cord")
        assert result["is_resolved"] is True

    def test_ambiguous_defaults_to_not_resolved(self):
        result = llm_interpret_troubleshooting_response("hmm I'm not sure", "Check the power cord")
        assert result["is_resolved"] is False

    def test_empty_input(self):
        result = llm_interpret_troubleshooting_response("", "Check the power cord")
        assert result["is_resolved"] is False
