import json
import re
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


# Regex for extracting email from LLM output
_EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Valid TLDs for email validation
_VALID_TLDS = {'.com', '.net', '.org', '.edu', '.gov', '.io', '.co', '.uk', '.ca', '.in'}

# Number word to digit mapping
_NUMBER_WORDS = {
    'zero': '0', 'oh': '0', 'o': '0',
    'one': '1', 'won': '1',
    'two': '2', 'to': '2', 'too': '2',
    'three': '3', 'tree': '3',
    'four': '4', 'for': '4', 'fore': '4',
    'five': '5',
    'six': '6', 'sicks': '6',
    'seven': '7',
    'eight': '8', 'ate': '8',
    'nine': '9', 'niner': '9',
}


def _normalize_speech_for_email(speech_text: str) -> str:
    """
    Pre-process speech-to-text before sending to LLM for email extraction.
    Handles common STT patterns deterministically.
    """
    text = speech_text.lower().strip()
    
    # Remove common filler words/phrases
    fillers = ['my email is', 'my email address is', 'its', "it's", 'yeah', 'yes', 
               'sure', 'um', 'uh', 'like', 'so', 'okay', 'ok']
    for filler in fillers:
        text = text.replace(filler, ' ')
    
    # Normalize @ symbol patterns
    at_patterns = [
        (r'\bat\s+the\s+rate\b', ' @ '),
        (r'\bat\s+rate\b', ' @ '),
        (r'\ba\s+great\b', ' @ '),  # "a great" misheard as "at rate"
        (r'\bat\s+sign\b', ' @ '),
        (r'\bat\s+symbol\b', ' @ '),
        (r'\s+at\s+', ' @ '),
    ]
    for pattern, replacement in at_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Normalize domain patterns
    domain_patterns = [
        (r'\bg\s*mail\b', 'gmail'),
        (r'\bgee\s*mail\b', 'gmail'),
        (r'\bjmail\b', 'gmail'),
        (r'\byahoo\b', 'yahoo'),
        (r'\boutlook\b', 'outlook'),
        (r'\bhotmail\b', 'hotmail'),
        (r'\bicloud\b', 'icloud'),
    ]
    for pattern, replacement in domain_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Normalize "dot com", "dot net", etc.
    tld_patterns = [
        (r'\bdot\s*com\b', '.com'),
        (r'\bdot\s*net\b', '.net'),
        (r'\bdot\s*org\b', '.org'),
        (r'\bdot\s*edu\b', '.edu'),
        (r'\bdot\s*co\s*dot\s*uk\b', '.co.uk'),
        (r'\bdot\s*io\b', '.io'),
    ]
    for pattern, replacement in tld_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Convert number words to digits
    words = text.split()
    converted = []
    for word in words:
        # Clean punctuation from word for matching
        clean_word = re.sub(r'[.,;:!?]', '', word)
        if clean_word in _NUMBER_WORDS:
            converted.append(_NUMBER_WORDS[clean_word])
        else:
            converted.append(word)
    text = ' '.join(converted)
    
    # Collapse spaced single digits: "1 2 3" -> "123"
    text = re.sub(r'(\d)\s+(?=\d)', r'\1', text)
    
    # Collapse spaced single letters (likely spelling): "k a s i" -> "kasi"
    # But be careful not to collapse around @ or .
    def collapse_letters(match):
        letters = match.group(0)
        return re.sub(r'\s+', '', letters)
    
    # Match sequences of single letters separated by spaces (at least 3)
    text = re.sub(r'\b([a-z])(?:\s+[a-z]){2,}\b', collapse_letters, text)
    
    # Remove STT artifacts: periods between letters "k. a. s. i" -> "kasi"
    text = re.sub(r'\b([a-z])\.\s*(?=[a-z]\b)', r'\1', text)
    
    # Normalize "dot" in middle of email (actual period): "john dot smith" -> "john.smith"
    text = re.sub(r'\s+dot\s+', '.', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def llm_extract_email(speech_text: str) -> str | None:
    """
    Uses Gemini to extract an email address from Twilio speech-to-text output.
    
    Handles messy STT like:
    - "K. A s. I dot m. A j. J. I at gmail.com"
    - "john at the rate gmail dot com"
    - "my email is j o h n 1 2 3 at yahoo dot com"
    
    Returns:
        Extracted email string if found, None otherwise.
    """
    if not speech_text or not speech_text.strip():
        print("[LLM Email] Empty input")
        return None
    
    if not model:
        print("[LLM Email] No Gemini model available")
        return None
    
    # Step 1: Deterministic pre-processing
    normalized = _normalize_speech_for_email(speech_text)
    print(f"[LLM Email] Normalized: '{normalized}' from '{speech_text}'")
    
    try:
        # Step 2: LLM extraction with few-shot examples
        prompt = f"""Extract the email address from this speech-to-text transcription.

Rules:
- Combine spelled letters: "k a s i" = "kasi"
- Number words to digits: "one two three" = "123"
- "at" or "@" = @
- Always include full domain (.com, .net, etc.)

Examples:
1) "k a s i one two three @ gmail.com" -> kasi123@gmail.com
2) "john 9 9 9 @ yahoo.com" -> john999@yahoo.com
3) "m y n a m e @ outlook.com" -> myname@outlook.com
4) "kasi.majji 9 9 9 @ gmail.com" -> kasi.majji999@gmail.com

Reply with ONLY the email address, or "none" if you cannot form a valid email.

Transcription: {normalized}"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64}
        )
        raw_result = result.text.strip()
        
        print(f"[LLM Email] Raw LLM output: '{raw_result}'")
        
        # Step 3: Robust post-processing
        # Strip code fences if present
        if '```' in raw_result:
            raw_result = re.sub(r'```[a-z]*\n?', '', raw_result)
            raw_result = raw_result.replace('```', '').strip()
        
        # Remove any quotes
        raw_result = raw_result.strip('"\'')
        
        # Check for explicit "none" response
        if raw_result.lower() in ('none', 'n/a', 'invalid', 'no email'):
            print(f"[LLM Email] LLM returned none for: '{speech_text}'")
            return None
        
        # Extract email using regex (handles extra text from LLM)
        match = _EMAIL_REGEX.search(raw_result)
        if not match:
            # Try after removing spaces (LLM might have added spaces)
            raw_no_spaces = raw_result.replace(' ', '')
            match = _EMAIL_REGEX.search(raw_no_spaces)
        
        if not match:
            print(f"[LLM Email] No valid email pattern found in: '{raw_result}'")
            return None
        
        email = match.group(0).lower()
        
        # Validate TLD
        has_valid_tld = any(email.endswith(tld) for tld in _VALID_TLDS)
        if not has_valid_tld:
            print(f"[LLM Email] Rejected - invalid TLD: '{email}'")
            return None
        
        print(f"[LLM Email] Extracted: '{email}'")
        return email
        
    except Exception as e:
        print(f"[LLM Email Error] Extraction failed: {e}")
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
