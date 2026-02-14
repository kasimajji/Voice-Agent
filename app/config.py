import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Application base URL for external callbacks (Twilio webhooks, email links, etc.)
# Set this to your ngrok URL locally or your production domain in cloud deployments
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# Google Cloud credentials (service account JSON for STT/TTS)
GCP_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# TTS Configuration — Google WaveNet voices for natural-sounding speech
# Twilio supports Google WaveNet voices directly in <Say> element
# Available: Google.en-US-Wavenet-A (male), Google.en-US-Wavenet-C (female),
#            Google.en-US-Wavenet-D (male), Google.en-US-Wavenet-F (female)
TTS_VOICE = os.getenv("TTS_VOICE", "Google.en-US-Wavenet-F")

# STT Configuration
# Gather-based STT (fallback) — phone_call model for telephony audio
STT_SPEECH_MODEL = os.getenv("STT_SPEECH_MODEL", "phone_call")
STT_ENHANCED = True

# Google STT v2 streaming — used via Twilio Media Streams WebSocket
USE_STREAMING_STT = os.getenv("USE_STREAMING_STT", "true").lower() == "true"
STT_CONFIDENCE_THRESHOLD = float(os.getenv("STT_CONFIDENCE_THRESHOLD", "0.4"))
STT_SILENCE_TIMEOUT_MS = int(os.getenv("STT_SILENCE_TIMEOUT_MS", "800"))
STT_LANGUAGE_CODE = os.getenv("STT_LANGUAGE_CODE", "en-US")

# Use basic logging here since logging_config may not be loaded yet
_config_logger = logging.getLogger("voice_agent.config")

if not GEMINI_API_KEY:
    _config_logger.warning("GOOGLE_API_KEY is not set. LLM features will fall back to keyword-based logic.")

if APP_BASE_URL == "http://localhost:8000":
    _config_logger.warning("APP_BASE_URL is not set. Using default http://localhost:8000. Set APP_BASE_URL for production or ngrok.")


def get_base_url_from_request(request) -> str:
    """
    Derive the public base URL from the incoming request's Host header.
    This solves the ngrok URL drift problem: the twilio-config container updates
    Twilio with the live ngrok URL, but APP_BASE_URL may be stale.
    
    Falls back to APP_BASE_URL if Host header is missing.
    """
    host = request.headers.get("host", "")
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    
    if host:
        # ngrok and most reverse proxies set x-forwarded-proto
        scheme = forwarded_proto if forwarded_proto else "https"
        return f"{scheme}://{host}"
    
    return APP_BASE_URL
