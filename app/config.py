import os
from dotenv import load_dotenv

load_dotenv()

# Application base URL for external callbacks (Twilio webhooks, email links, etc.)
# Set this to your ngrok URL locally or your production domain in cloud deployments
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("[WARNING] GOOGLE_API_KEY is not set. LLM features will fall back to keyword-based logic.")

if APP_BASE_URL == "http://localhost:8000":
    print("[WARNING] APP_BASE_URL is not set. Using default http://localhost:8000. Set APP_BASE_URL for production or ngrok.")
