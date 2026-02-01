import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("[WARNING] GOOGLE_API_KEY is not set. LLM features will fall back to keyword-based logic.")
