import json
import re
import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL
from .logging_config import get_logger

logger = get_logger("llm")

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


def llm_extract_name(speech_text: str) -> str:
    """
    Extract customer name from speech using LLM.
    
    Handles various phrasings like:
    - "My name is John"
    - "I'm Sarah"
    - "This is Mike calling"
    - "John Smith"
    - "It's Kasim"
    
    Returns:
        First name only, title-cased. Returns "there" if no name found.
    """
    if not model or not speech_text.strip():
        return "there"
    
    # Simple regex-based extraction as primary method (more reliable)
    text = speech_text.strip()
    
    # Remove filler words
    filler_pattern = r'^(uh,?\s*|um,?\s*|yeah,?\s*|yes,?\s*|so,?\s*|well,?\s*|okay,?\s*|ok,?\s*|hey,?\s*|hi,?\s*)+'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE).strip()
    
    # Try to extract name after common prefixes
    name_patterns = [
        r"my name is\s+([A-Za-z]+)",
        r"i'm\s+([A-Za-z]+)",
        r"i am\s+([A-Za-z]+)",
        r"this is\s+([A-Za-z]+)",
        r"it's\s+([A-Za-z]+)",
        r"call me\s+([A-Za-z]+)",
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).title()
            logger.debug(f"Name extracted via pattern: '{name}' from '{speech_text}'")
            return name
    
    # If no pattern matched, take first capitalized word that looks like a name
    words = text.split()
    for word in words:
        clean_word = word.strip(".,!?'\"")
        # Skip common non-name words
        skip_words = {"hey", "hi", "hello", "yes", "no", "yeah", "um", "uh", "well", "so", "okay", "ok", "good", "fine", "great", "thanks", "thank", "you", "the", "a", "an", "is", "am", "are", "my", "name", "i", "i'm", "it's", "this"}
        if clean_word.lower() not in skip_words and len(clean_word) >= 2 and clean_word.isalpha():
            name = clean_word.title()
            logger.debug(f"Name extracted from first valid word: '{name}' from '{speech_text}'")
            return name
    
    logger.debug(f"No valid name found in: '{speech_text}', using 'there'")
    return "there"


def llm_interpret_troubleshooting_response(speech_text: str, troubleshooting_step: str) -> dict:
    """
    Uses Gemini to intelligently interpret customer response during troubleshooting.
    Understands context and nuance to determine if the issue is resolved or persists.
    
    Args:
        speech_text: Customer's response
        troubleshooting_step: The troubleshooting instruction that was given
    
    Returns:
        dict with:
        - is_resolved: bool (True if issue appears fixed)
        - confidence: str ("high", "medium", "low")
        - interpretation: str (what the customer actually meant)
    """
    if not speech_text or not speech_text.strip():
        return {"is_resolved": False, "confidence": "low", "interpretation": "No response"}
    
    if not model:
        # Fallback: simple keyword matching
        text_lower = speech_text.lower()
        positive_words = ["yes", "yeah", "fixed", "working", "resolved", "good now", "helped"]
        negative_words = ["no", "still", "not working", "same issue", "didn't help", "worse"]
        
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_words)
        
        if has_negative:
            return {"is_resolved": False, "confidence": "medium", "interpretation": speech_text}
        elif has_positive:
            return {"is_resolved": True, "confidence": "medium", "interpretation": speech_text}
        else:
            return {"is_resolved": False, "confidence": "low", "interpretation": speech_text}
    
    try:
        prompt = f"""You are helping interpret a customer's response during appliance troubleshooting.

Troubleshooting step given: "{troubleshooting_step}"

Customer's response: "{speech_text}"

Analyze the response and determine:
1. Is the issue RESOLVED (fixed/working) or PERSISTS (still broken/not working)?
2. What did the customer actually mean?

IMPORTANT: Look for context clues:
- "I checked it, it's at max cooling" means they checked but issue PERSISTS (not cooling despite max setting)
- "It's good" could mean either fixed OR already checked (use context)
- "Still having the issue" clearly means PERSISTS
- "That worked!" clearly means RESOLVED
- "No change" means PERSISTS

Respond in JSON format:
{{
  "is_resolved": true/false,
  "confidence": "high/medium/low",
  "interpretation": "brief explanation of what customer meant"
}}

Examples:

Input: "No, still having an issue"
Output: {{"is_resolved": false, "confidence": "high", "interpretation": "Issue persists"}}

Input: "I checked the dial, it's at max cooling"
Output: {{"is_resolved": false, "confidence": "high", "interpretation": "Customer confirmed setting is correct but issue persists"}}

Input: "Yes, that fixed it!"
Output: {{"is_resolved": true, "confidence": "high", "interpretation": "Issue is resolved"}}

Input: "The door is good"
Output: {{"is_resolved": false, "confidence": "medium", "interpretation": "Door seal is fine but issue persists"}}

Now analyze:"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 128}
        )
        
        # Handle response
        try:
            raw_result = result.text.strip()
        except (ValueError, AttributeError):
            if hasattr(result, 'candidates') and result.candidates:
                try:
                    raw_result = result.candidates[0].content.parts[0].text.strip()
                except (AttributeError, IndexError):
                    return {"is_resolved": False, "confidence": "low", "interpretation": speech_text}
            else:
                return {"is_resolved": False, "confidence": "low", "interpretation": speech_text}
        
        # Parse JSON response
        raw_result = raw_result.strip()
        if '```json' in raw_result:
            raw_result = re.sub(r'```json\n?', '', raw_result)
            raw_result = raw_result.replace('```', '').strip()
        elif '```' in raw_result:
            raw_result = re.sub(r'```[a-z]*\n?', '', raw_result)
            raw_result = raw_result.replace('```', '').strip()
        
        # Ensure we have valid JSON
        if not raw_result or not raw_result.startswith('{'):
            logger.warning(f"Invalid JSON format: '{raw_result}', using fallback")
            raise ValueError("Invalid JSON format")
        
        parsed = json.loads(raw_result)
        logger.debug(f"Interpreted '{speech_text}' as: {parsed}")
        return parsed
        
    except Exception as e:
        logger.error(f"Troubleshoot interpretation error: {e}")
        # Fallback to simple keyword matching
        text_lower = speech_text.lower()
        if any(word in text_lower for word in ["no", "still", "not working", "didn't help"]):
            return {"is_resolved": False, "confidence": "medium", "interpretation": speech_text}
        elif any(word in text_lower for word in ["yes", "fixed", "working", "helped"]):
            return {"is_resolved": True, "confidence": "medium", "interpretation": speech_text}
        else:
            return {"is_resolved": False, "confidence": "low", "interpretation": speech_text}


def llm_extract_name(speech_text: str) -> str | None:
    """
    Uses Gemini to extract a person's name from speech input.
    Filters out noise, random words, and non-name responses.
    
    Args:
        speech_text: The transcribed speech from Twilio
    
    Returns:
        Extracted name if valid, None if noise/invalid
    """
    if not speech_text or not speech_text.strip():
        return None
    
    if not model:
        # Fallback: simple extraction without LLM
        text = speech_text.strip()
        for prefix in ["my name is ", "i'm ", "this is ", "it's ", "i am ", "hey ", "hi "]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        name = text.split()[0] if text.split() else None
        return name if name and len(name) > 1 else None
    
    try:
        prompt = f"""Extract the person's name from this speech transcription.

Rules:
1. Return ONLY the first name (e.g., "John", "Sarah", "Mike")
2. If the input is noise, random words, or not a name, return "none"
3. Common patterns: "My name is John", "I'm Sarah", "This is Mike", or just "John"
4. Ignore filler words like "uh", "um", "whatever", "just", etc.

Examples:
- "My name is John Smith" -> John
- "I'm Sarah" -> Sarah
- "Mike" -> Mike
- "Whatever" -> none
- "Uh, just testing" -> none
- "Background noise" -> none
- "I'm good" -> none

Transcription: {speech_text}

Name:"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 32}
        )
        
        # Handle both simple and multi-part responses
        try:
            raw_result = result.text.strip()
        except (ValueError, AttributeError) as e:
            logger.debug(f"Multi-part response: {e}")
            if hasattr(result, 'parts') and result.parts:
                text_parts = []
                for part in result.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    raw_result = ''.join(text_parts).strip()
                else:
                    logger.debug("No text in parts")
                    return None
            elif hasattr(result, 'candidates') and result.candidates and len(result.candidates) > 0:
                try:
                    raw_result = result.candidates[0].content.parts[0].text.strip()
                except (AttributeError, IndexError) as ex:
                    logger.debug(f"Cannot extract from candidates: {ex}, using fallback")
                    # Use simple fallback extraction
                    text = speech_text.strip()
                    for prefix in ["my name is ", "i'm ", "this is ", "it's ", "i am ", "hey ", "hi "]:
                        if text.lower().startswith(prefix):
                            text = text[len(prefix):]
                            break
                    name = text.split()[0] if text.split() else None
                    if name:
                        name = name.strip('.,!?;:"\'')
                        if len(name) > 1 and name.isalpha():
                            name = name.capitalize()
                            logger.debug(f"Fallback extracted name: '{name}'")
                            return name
                    return None
            else:
                logger.debug("No candidates in response, using fallback")
                # Use simple fallback extraction
                text = speech_text.strip()
                for prefix in ["my name is ", "i'm ", "this is ", "it's ", "i am ", "hey ", "hi "]:
                    if text.lower().startswith(prefix):
                        text = text[len(prefix):]
                        break
                name = text.split()[0] if text.split() else None
                if name:
                    name = name.strip('.,!?;:"\'')
                    if len(name) > 1 and name.isalpha():
                        name = name.capitalize()
                        logger.debug(f"Fallback extracted name: '{name}'")
                        return name
                return None
        
        logger.debug(f"Raw LLM output: '{raw_result}'")
        
        # Clean up result
        raw_result = raw_result.strip('"\' ').lower()
        
        # Check for explicit "none" response
        if raw_result in ('none', 'n/a', 'invalid', 'noise', 'no name'):
            logger.debug(f"LLM returned none for: '{speech_text}'")
            return None
        
        # Validate it looks like a name (alphabetic, reasonable length)
        if raw_result and raw_result.isalpha() and 2 <= len(raw_result) <= 20:
            name = raw_result.capitalize()
            logger.debug(f"Extracted name: '{name}' from '{speech_text}'")
            return name
        
        logger.debug(f"Invalid name format: '{raw_result}' (isalpha={raw_result.isalpha() if raw_result else False}, len={len(raw_result) if raw_result else 0})")
        return None
        
    except Exception as e:
        logger.error(f"Name extraction failed: {e}")
        # Fallback to simple extraction
        text = speech_text.strip()
        for prefix in ["my name is ", "i'm ", "this is ", "it's ", "i am "]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        name = text.split()[0] if text.split() else None
        return name if name and len(name) > 1 and name.isalpha() else None


def llm_is_appliance_related(user_text: str) -> bool:
    """
    Checks if the user's input is related to home appliances.
    Uses keyword/brand detection first, then LLM as backup.
    Returns True if appliance-related, False otherwise.
    """
    # First check for brand names or appliance keywords (fast, handles STT errors)
    if _contains_appliance_hint(user_text):
        logger.debug(f"Brand/keyword detected in: '{user_text}' -> True")
        return True
    
    if not model:
        logger.debug("No Gemini model available, assuming appliance-related")
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
        logger.debug(f"Appliance relevance check: '{user_text}' -> {is_related}")
        return is_related
        
    except Exception as e:
        logger.error(f"Appliance relevance check failed: {e}")
        return True  # Default to True on error to avoid blocking flow


def llm_classify_appliance(user_text: str) -> str | None:
    """
    Uses Gemini to classify the appliance type from user text.
    Returns one of: washer, dryer, refrigerator, dishwasher, oven, hvac, or None.
    """
    if not model:
        logger.debug("No Gemini model available, skipping LLM classification")
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
        
        logger.debug(f"Appliance classification result: {appliance}")
        
        if appliance in VALID_APPLIANCES:
            return appliance if appliance != "other" else None
        return None
        
    except Exception as e:
        logger.error(f"Appliance classification failed: {e}")
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
    
    Key fix: Aggressively remove periods/commas between letters FIRST to prevent
    LLM hallucination/truncation on inputs like "S. H. I. N. Y."
    """
    text = speech_text.lower().strip()
    
    # STEP 1: Protect TLDs BEFORE aggressive period removal
    # Replace .com, .net, etc. with placeholders to preserve them
    tld_placeholders = {
        '.com': '___DOTCOM___',
        '.net': '___DOTNET___',
        '.org': '___DOTORG___',
        '.edu': '___DOTEDU___',
        '.io': '___DOTIO___',
        '.co.uk': '___DOTCOUK___',
    }
    for tld, placeholder in tld_placeholders.items():
        text = text.replace(tld, placeholder)
    
    # STEP 2: Remove periods and commas that sit between letters/spaces
    # This turns "s. h. i. n. y." into "s  h  i  n  y " before any other processing
    # Prevents LLM from treating periods as sentence boundaries
    text = re.sub(r'(?<=[a-z0-9\s])[.,](?=[a-z0-9\s]|$)', ' ', text)
    
    # STEP 3: Restore TLD placeholders
    for tld, placeholder in tld_placeholders.items():
        text = text.replace(placeholder, tld)
    
    # Remove common filler words/phrases (use word boundaries to avoid partial matches)
    filler_patterns = [
        r'\bmy email is\b', r'\bmy email address is\b', r'\bits\b', r"\bit's\b", 
        r'\byeah\b', r'\byes\b', r'\bsure\b', r'\bum\b', r'\buh\b', 
        r'\blike\b', r'\bso\b', r'\bokay\b', r'\bok\b'
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Normalize @ symbol patterns (order matters - more specific first)
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
    
    # Normalize "dot com", "dot net", etc. FIRST (before letter collapsing)
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
    
    # Normalize "dot" in middle of email (actual period): "john dot smith" -> "john.smith"
    text = re.sub(r'\s+dot\s+', '.', text)
    
    # Convert number words to digits
    words = text.split()
    converted = []
    for word in words:
        clean_word = re.sub(r'[.,;:!?]', '', word)
        if clean_word in _NUMBER_WORDS:
            converted.append(_NUMBER_WORDS[clean_word])
        else:
            converted.append(word)
    text = ' '.join(converted)
    
    # Collapse spaced single digits: "1 2 3" -> "123"
    text = re.sub(r'(\d)\s+(?=\d)', r'\1', text)
    
    # Remove any remaining punctuation between letters (cleanup pass)
    text = re.sub(r'([a-z])\s*[,;:!?]\s*(?=[a-z])', r'\1 ', text)
    
    # Collapse spaced single letters (likely spelling): "k a s i" -> "kasi"
    # Match sequences of single letters separated by spaces (at least 2)
    # But DON'T collapse if there's a . or @ nearby (preserve email structure)
    def collapse_letters(match):
        letters = match.group(0)
        if '@' in letters or '.' in letters:
            return letters
        return re.sub(r'\s+', '', letters)
    
    text = re.sub(r'\b([a-z])(?:\s+[a-z]){1,}\b', collapse_letters, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def llm_extract_email(speech_text: str) -> str | None:
    """
    Extract email address from Twilio speech-to-text output using deterministic processing.
    
    NO LLM CALL - pure regex-based extraction on normalized text.
    The normalization handles all the heavy lifting:
    - Spelled out letters: "k a s i" -> "kasi"
    - @ patterns: "at", "at the rate" -> "@"
    - Domain patterns: "gmail", "yahoo" -> normalized
    - TLD patterns: "dot com" -> ".com"
    
    Returns:
        Extracted email string if found, None otherwise.
    """
    if not speech_text or not speech_text.strip():
        logger.debug("Email extract: Empty input")
        return None
    
    # Step 1: Deterministic pre-processing (handles all STT artifacts)
    normalized = _normalize_speech_for_email(speech_text)
    logger.debug(f"Email normalized: '{normalized}' from '{speech_text}'")
    
    # Step 2: Build email from normalized text
    # The normalized text should look like: "shinyangelinajalli @ gmail.com"
    # We need to remove spaces around @ and extract the email
    
    # Remove spaces around @
    email_candidate = re.sub(r'\s*@\s*', '@', normalized)
    
    # Remove any trailing punctuation
    email_candidate = email_candidate.rstrip('.,;:!?')
    
    logger.debug(f"Email candidate after cleanup: '{email_candidate}'")
    
    # Step 3: Extract email using regex
    match = _EMAIL_REGEX.search(email_candidate)
    
    if not match:
        # Try removing all remaining spaces (in case there are still some)
        email_no_spaces = email_candidate.replace(' ', '')
        match = _EMAIL_REGEX.search(email_no_spaces)
        if match:
            logger.debug(f"Email found after removing spaces: '{email_no_spaces}'")
    
    if not match:
        logger.debug(f"No valid email pattern found in: '{email_candidate}'")
        return None
    
    email = match.group(0).lower()
    
    # Step 4: Validate TLD
    has_valid_tld = any(email.endswith(tld) for tld in _VALID_TLDS)
    if not has_valid_tld:
        logger.debug(f"Email rejected - invalid TLD: '{email}'")
        return None
    
    logger.info(f"Email extracted: '{email}'")
    return email


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
        logger.debug("No Gemini model available, using fallback for symptoms")
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
        
        logger.debug(f"Symptom extraction raw result: {raw}")
        
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
        
        logger.debug(f"Symptom extraction parsed: {extracted}")
        return extracted
        
    except Exception as e:
        logger.error(f"Symptom extraction failed: {e}")
        return fallback
