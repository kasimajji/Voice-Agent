"""Tests for app.twilio_routes module — TTS, email readback, helpers, and route handlers."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# ── Helper function tests (no DB needed) ───────────────────────────────

class TestSpeakEmailNaturally:
    def setup_method(self):
        from app.twilio_routes import speak_email_naturally
        self.speak = speak_email_naturally

    def test_basic_email(self):
        result = self.speak("john@gmail.com")
        assert "john at gmail dot com" in result
        assert "spell" in result.lower()
        assert "correct" in result.lower()

    def test_email_with_dot_in_username(self):
        result = self.speak("kasi.majji@gmail.com")
        assert "kasi dot majji at gmail dot com" in result

    def test_email_with_underscore(self):
        result = self.speak("kasi_majji@yahoo.com")
        assert "underscore" in result

    def test_empty_email(self):
        assert self.speak("") == ""

    def test_none_email(self):
        assert self.speak(None) == ""

    def test_includes_spelling(self):
        result = self.speak("ab@c.com")
        # Should contain individual letters in the spelled-out portion
        assert "a, b, at, c, dot, c, o, m" in result


class TestSpellEmailSlow:
    def setup_method(self):
        from app.twilio_routes import _spell_email_slow
        self.spell = _spell_email_slow

    def test_basic_email(self):
        result = self.spell("a@b.com")
        assert result == "a, at, b, dot, c, o, m"

    def test_special_chars(self):
        result = self.spell("a_b-c+d@e.com")
        assert "underscore" in result
        assert "dash" in result
        assert "plus" in result

    def test_empty(self):
        assert self.spell("") == ""

    def test_none(self):
        assert self.spell(None) == ""


class TestIsYesNoResponse:
    def setup_method(self):
        from app.twilio_routes import is_yes_response, is_no_response
        self.is_yes = is_yes_response
        self.is_no = is_no_response

    @pytest.mark.parametrize("text", ["yes", "yeah", "yep", "correct", "ok", "okay", "that's right"])
    def test_yes_responses(self, text):
        assert self.is_yes(text) is True

    @pytest.mark.parametrize("text", ["no", "nope", "wrong", "incorrect", "try again"])
    def test_no_responses(self, text):
        assert self.is_no(text) is True

    def test_yes_not_no(self):
        assert self.is_yes("yes") is True
        assert self.is_no("yes") is False

    def test_no_not_yes(self):
        assert self.is_no("no") is True
        assert self.is_yes("no") is False


class TestCreateSsmlSay:
    def test_returns_say_object(self):
        from app.twilio_routes import create_ssml_say
        say = create_ssml_say("Hello world")
        xml = str(say)
        assert "Hello world" in xml

    def test_uses_neural_voice(self):
        from app.twilio_routes import create_ssml_say
        from app.config import TTS_VOICE
        say = create_ssml_say("test")
        xml = str(say)
        assert TTS_VOICE in xml


class TestBuildGather:
    def test_gather_has_speech_model(self):
        from twilio.twiml.voice_response import VoiceResponse
        from app.twilio_routes import _build_gather
        from app.config import STT_SPEECH_MODEL

        resp = VoiceResponse()
        _build_gather(resp, "https://example.com/continue", timeout=5)
        xml = str(resp)
        assert STT_SPEECH_MODEL in xml

    def test_gather_barge_in_false(self):
        from twilio.twiml.voice_response import VoiceResponse
        from app.twilio_routes import _build_gather

        resp = VoiceResponse()
        _build_gather(resp, "https://example.com/continue")
        xml = str(resp)
        assert 'bargeIn="false"' in xml

    def test_gather_with_hints(self):
        from twilio.twiml.voice_response import VoiceResponse
        from app.twilio_routes import _build_gather

        resp = VoiceResponse()
        _build_gather(resp, "https://example.com/continue", hints="gmail.com, yahoo.com")
        xml = str(resp)
        assert "gmail.com" in xml


class TestGetContinueUrl:
    def test_uses_host_header(self):
        from app.twilio_routes import _get_continue_url

        mock_request = MagicMock()
        mock_request.headers = {
            "host": "abc.ngrok-free.app",
            "x-forwarded-proto": "https",
        }
        url = _get_continue_url(mock_request)
        assert url == "https://abc.ngrok-free.app/twilio/voice/continue"

    def test_fallback_to_app_base_url(self):
        from app.twilio_routes import _get_continue_url
        from app.config import APP_BASE_URL

        mock_request = MagicMock()
        mock_request.headers = {}
        url = _get_continue_url(mock_request)
        assert url == f"{APP_BASE_URL}/twilio/voice/continue"


# ── Route handler tests (mock DB) ─────────────────────────────────────

class TestVoiceEntryRoute:
    """Test the /voice entry endpoint."""

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_call_start")
    def test_voice_entry_returns_twiml(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {"step": "greet_ask_name", "customer_phone": None, "no_input_attempts": 0}

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice",
            data={"CallSid": "CA123", "From": "+15551234567", "To": "+15559876543"},
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        body = resp.text
        assert "Sears Home Services" in body
        assert "Gather" in body

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_call_start")
    def test_voice_entry_sets_step(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {"step": "greet_ask_name", "customer_phone": None, "no_input_attempts": 0}

        client = TestClient(app)
        client.post("/twilio/voice", data={"CallSid": "CA123", "From": "+1", "To": "+2"})

        # update_state should be called with step = greet_ask_name
        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "greet_ask_name"


class TestVoiceContinueGreetAskName:
    """Test the greet_ask_name step."""

    @patch("app.twilio_routes.llm_extract_name", return_value="John")
    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_name_extraction_moves_to_understand_need(self, mock_log, mock_update, mock_get, mock_name):
        from app.main import app
        mock_get.return_value = {
            "step": "greet_ask_name",
            "customer_phone": "+1",
            "no_input_attempts": 0,
            "customer_name": None,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": "My name is John"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "John" in body
        assert "help you today" in body

        # State should move to understand_need
        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "understand_need"
        assert call_args[0][1]["customer_name"] == "John"


class TestVoiceContinueUnderstandNeed:
    """Test the autonomous intent detection step."""

    @patch("app.twilio_routes.llm_analyze_customer_intent")
    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_schedule_intent_skips_to_zip(self, mock_log, mock_update, mock_get, mock_intent):
        from app.main import app
        mock_get.return_value = {
            "step": "understand_need",
            "customer_name": "John",
            "no_input_attempts": 0,
        }
        mock_intent.return_value = {
            "intent": "schedule_technician",
            "appliance_type": "washer",
            "symptoms": None,
            "wants_scheduling": True,
            "has_full_description": False,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": "I want to schedule a technician for my washer"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "ZIP code" in body

        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "collect_zip"

    @patch("app.twilio_routes.llm_analyze_customer_intent")
    @patch("app.twilio_routes.llm_extract_symptoms")
    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_full_description_offers_troubleshoot_or_schedule(self, mock_log, mock_update, mock_get, mock_symptoms, mock_intent):
        from app.main import app
        mock_get.return_value = {
            "step": "understand_need",
            "customer_name": "Jane",
            "no_input_attempts": 0,
        }
        mock_intent.return_value = {
            "intent": "describe_problem",
            "appliance_type": "refrigerator",
            "symptoms": "Fridge not cooling, ice buildup in freezer",
            "wants_scheduling": False,
            "has_full_description": True,
        }
        mock_symptoms.return_value = {
            "symptom_summary": "Fridge not cooling, ice buildup in freezer",
            "error_codes": [],
            "is_urgent": False,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA456", "SpeechResult": "My fridge is not cooling and there's ice buildup in the freezer"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "troubleshooting" in body.lower() or "schedule" in body.lower()

    @patch("app.twilio_routes.llm_analyze_customer_intent")
    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_appliance_only_asks_for_symptoms(self, mock_log, mock_update, mock_get, mock_intent):
        from app.main import app
        mock_get.return_value = {
            "step": "understand_need",
            "customer_name": "Bob",
            "no_input_attempts": 0,
        }
        mock_intent.return_value = {
            "intent": "describe_problem",
            "appliance_type": "dryer",
            "symptoms": None,
            "wants_scheduling": False,
            "has_full_description": False,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA789", "SpeechResult": "My dryer"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "more about" in body.lower() or "what's happening" in body.lower()

        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "ask_symptoms"


class TestVoiceContinueNoInput:
    """Test no-input handling."""

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_no_input_prompts_retry(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {
            "step": "greet_ask_name",
            "no_input_attempts": 0,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": ""},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "didn't hear" in body.lower()

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_three_no_inputs_falls_back_to_scheduling(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {
            "step": "greet_ask_name",
            "no_input_attempts": 3,
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": ""},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "ZIP code" in body


class TestVoiceContinueConfirmZip:
    """Test the new ZIP confirmation step."""

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_zip_captured_asks_confirmation(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {
            "step": "collect_zip",
            "no_input_attempts": 0,
            "zip_attempts": 0,
            "customer_name": "John",
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": "60601"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "correct" in body.lower()

        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "confirm_zip"
        assert call_args[0][1]["zip_code"] == "60601"

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_zip_confirmed_moves_to_time_pref(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {
            "step": "confirm_zip",
            "zip_code": "60601",
            "no_input_attempts": 0,
            "customer_name": "John",
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": "yes"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "morning" in body.lower() or "afternoon" in body.lower()

        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "collect_time_pref"

    @patch("app.twilio_routes.get_state")
    @patch("app.twilio_routes.update_state")
    @patch("app.twilio_routes.log_conversation")
    def test_zip_rejected_asks_again(self, mock_log, mock_update, mock_get):
        from app.main import app
        mock_get.return_value = {
            "step": "confirm_zip",
            "zip_code": "60601",
            "no_input_attempts": 0,
            "customer_name": "John",
        }

        client = TestClient(app)
        resp = client.post(
            "/twilio/voice/continue",
            data={"CallSid": "CA123", "SpeechResult": "no that's wrong"},
        )
        assert resp.status_code == 200

        call_args = mock_update.call_args
        assert call_args[0][1]["step"] == "collect_zip"
        assert call_args[0][1]["zip_code"] is None
