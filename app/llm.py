import json
import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL

model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

# Generation config optimized for voice applications (fast, concise responses)
GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_output_tokens": 256,
}

VALID_APPLIANCES = {"washer", "dryer", "refrigerator", "dishwasher", "oven", "hvac", "other"}

# Common appliance brand names - if mentioned, assume appliance-related
APPLIANCE_BRANDS = {
    "samsung", "lg", "whirlpool", "ge", "general electric", "maytag", "frigidaire",
    "kenmore", "bosch", "kitchenaid", "electrolux", "amana", "hotpoint", "haier",
    "thermador", "viking", "sub-zero", "subzero", "wolf", "miele", "speed queen",
    "carrier", "trane", "lennox", "rheem", "goodman", "daikin", "mitsubishi"
}

# Appliance-related keywords (in case STT mangles the exact word)
APPLIANCE_KEYWORDS = {
    "washer", "washing", "dryer", "drying", "fridge", "refrigerator", "freezer",
    "dishwasher", "dishes", "oven", "stove", "range", "cooktop", "microwave",
    "hvac", "heating", "cooling", "air conditioner", "ac", "furnace", "heat pump"
}


def _contains_appliance_hint(text: str) -> bool:
    """Check if text contains brand names or appliance keywords."""
    text_lower = text.lower()
    for brand in APPLIANCE_BRANDS:
        if brand in text_lower:
            return True
    for keyword in APPLIANCE_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def llm_is_appliance_related(user_text: str) -> bool:
    """
    Checks if the user's input is related to home appliances.
    Uses keyword/brand detection first, then LLM as backup.
    Returns True if appliance-related, False otherwise.
    """
    # First check for brand names or appliance keywords (fast, handles STT errors)
    if _contains_appliance_hint(user_text):
        print(f"[Validation] Brand/keyword detected in: '{user_text}' -> True")
        return True
    
    if not model:
        print("[LLM] No Gemini model available, assuming appliance-related")
        return True
    
    try:
        prompt = f"""You are a classification assistant for a home appliance service company.
Determine if the user's message is related to home appliances (washer, dryer, refrigerator, dishwasher, oven, HVAC, etc.).

Reply with ONLY "yes" or "no" (lowercase, no extra text).
- "yes" if the message mentions or implies a home appliance
- "no" if it's unrelated (random words, greetings without context, off-topic questions)

User message:
{user_text}"""

        result = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG
        )
        answer = result.text.strip().lower()
        
        is_related = answer == "yes" or answer.startswith("yes")
        print(f"[LLM] Appliance relevance check: '{user_text}' -> {is_related}")
        return is_related
        
    except Exception as e:
        print(f"[LLM Error] Appliance relevance check failed: {e}")
        return True  # Default to True on error to avoid blocking flow


def llm_classify_appliance(user_text: str) -> str | None:
    """
    Uses Gemini to classify the appliance type from user text.
    Returns one of: washer, dryer, refrigerator, dishwasher, oven, hvac, or None.
    """
    if not model:
        print("[LLM] No Gemini model available, skipping LLM classification")
        return None
    
    try:
        prompt = f"""You are a classification assistant. From the user text, identify the APPLIANCE TYPE only.
Valid answers: washer, dryer, refrigerator, dishwasher, oven, hvac, other.
Reply with just one of these words in lowercase, with no extra text.

User text:
{user_text}"""

        result = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG
        )
        appliance = result.text.strip().lower()
        
        print(f"[LLM] Appliance classification result: {appliance}")
        
        if appliance in VALID_APPLIANCES:
            return appliance if appliance != "other" else None
        return None
        
    except Exception as e:
        print(f"[LLM Error] Appliance classification failed: {e}")
        return None


def llm_extract_symptoms(user_text: str) -> dict:
    """
    Uses Gemini to extract structured symptom information from user text.
    Returns dict with: symptom_summary, error_codes, is_urgent.
    """
    fallback = {
        "symptom_summary": user_text,
        "error_codes": [],
        "is_urgent": False
    }
    
    if not model:
        print("[LLM] No Gemini model available, using fallback for symptoms")
        return fallback
    
    try:
        prompt = f"""You are a home appliance service assistant.
From the caller's description, extract structured information.

Always respond in valid JSON with exactly these keys:
- "symptom_summary": string (a concise 1-2 sentence summary of the problem)
- "error_codes": list of strings (error codes like "E23", "F21", etc.)
- "is_urgent": boolean (true if safety issue, flooding, fire risk, gas smell, etc.)

If there are no obvious error codes, use an empty list for "error_codes".
If it does not sound urgent, use false for "is_urgent".

Caller description:
{user_text}"""

        result = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG
        )
        raw = result.text.strip()
        
        print(f"[LLM] Symptom extraction raw result: {raw}")
        
        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json"):
                    in_json = True
                    continue
                if line.startswith("```"):
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            raw = "\n".join(json_lines)
        
        data = json.loads(raw)
        
        extracted = {
            "symptom_summary": data.get("symptom_summary") or user_text,
            "error_codes": data.get("error_codes") or [],
            "is_urgent": bool(data.get("is_urgent"))
        }
        
        print(f"[LLM] Symptom extraction parsed: {extracted}")
        return extracted
        
    except Exception as e:
        print(f"[LLM Error] Symptom extraction failed: {e}")
        return fallback
