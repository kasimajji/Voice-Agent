import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Application base URL for external callbacks (Twilio webhooks, email links, etc.)
# Set this to your ngrok URL locally or your production domain in cloud deployments
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# TTS Configuration — Polly Neural voices sound significantly more natural than standard
# Available neural voices: Polly.Joanna-Neural, Polly.Matthew-Neural, Polly.Salli-Neural
TTS_VOICE = os.getenv("TTS_VOICE", "Polly.Joanna-Neural")

# STT Configuration — phone_call model is optimized for telephony audio
STT_SPEECH_MODEL = "phone_call"
STT_ENHANCED = True

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
