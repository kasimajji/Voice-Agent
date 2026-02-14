"""Tests for app.config module."""
import os
import pytest
from unittest.mock import MagicMock


def test_tts_voice_default():
    """TTS_VOICE should default to Polly.Joanna-Neural."""
    from app.config import TTS_VOICE
    assert "Neural" in TTS_VOICE or TTS_VOICE == os.getenv("TTS_VOICE", "Polly.Joanna-Neural")


def test_stt_speech_model():
    """STT_SPEECH_MODEL should be 'phone_call'."""
    from app.config import STT_SPEECH_MODEL
    assert STT_SPEECH_MODEL == "phone_call"


def test_get_base_url_from_request_with_host():
    """get_base_url_from_request should derive URL from Host header."""
    from app.config import get_base_url_from_request

    mock_request = MagicMock()
    mock_request.headers = {
        "host": "abc123.ngrok-free.app",
        "x-forwarded-proto": "https",
    }
    result = get_base_url_from_request(mock_request)
    assert result == "https://abc123.ngrok-free.app"


def test_get_base_url_from_request_no_forwarded_proto():
    """Should default to https when x-forwarded-proto is missing."""
    from app.config import get_base_url_from_request

    mock_request = MagicMock()
    mock_request.headers = {"host": "example.com"}
    result = get_base_url_from_request(mock_request)
    assert result == "https://example.com"


def test_get_base_url_from_request_no_host():
    """Should fall back to APP_BASE_URL when Host header is missing."""
    from app.config import get_base_url_from_request, APP_BASE_URL

    mock_request = MagicMock()
    mock_request.headers = {}
    result = get_base_url_from_request(mock_request)
    assert result == APP_BASE_URL
