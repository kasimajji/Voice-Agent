import json
import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL

model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

VALID_APPLIANCES = {"washer", "dryer", "refrigerator", "dishwasher", "oven", "hvac", "other"}


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

        result = model.generate_content(prompt)
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

        result = model.generate_content(prompt)
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
