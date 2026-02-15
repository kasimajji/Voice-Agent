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
    Extract customer name from speech - NEVER returns None.
    
    Uses LLM to intelligently decode what the customer said, including:
    - Phonetically similar names (Cassie/Kasi, John/Jon)
    - Names with accents or unusual pronunciations
    - Names mangled by speech-to-text
    
    Returns:
        First name only, title-cased. Always returns a name, never "there" or None.
    """
    if not speech_text.strip():
        return "Friend"
    
    text = speech_text.strip()
    
    # Remove filler words
    filler_pattern = r'^(uh,?\s*|um,?\s*|yeah,?\s*|yes,?\s*|so,?\s*|well,?\s*|okay,?\s*|ok,?\s*|hey,?\s*|hi,?\s*)+'
    text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE).strip()
    
    # Try regex patterns first
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
    
    # Use LLM to decode the name phonetically
    if model:
        try:
            prompt = f"""Extract the person's first name from this phone call greeting.

Speech: "{speech_text}"

Rules:
- Return ONLY the first name, nothing else
- Look for patterns like "my name is X", "this is X", "I'm X"
- Ignore greetings like "hey", "hi", "how are you"
- If multiple names possible, pick the one after "name is" or "I'm"

Just the name:"""

            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 10}
            )
            
            # Handle multi-part responses safely
            raw_result = ""
            try:
                raw_result = response.text.strip()
            except Exception as e:
                logger.debug(f"Multi-part response: {e}")
                try:
                    if response.candidates:
                        for candidate in response.candidates:
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        raw_result = part.text.strip()
                                        break
                                if raw_result:
                                    break
                except Exception as e2:
                    logger.debug(f"Cannot extract from candidates: {e2}")
            
            if raw_result:
                logger.debug(f"Raw LLM output: '{raw_result}'")
                name = raw_result.strip('"\'.,!?').title()
                
                # Validate result
                invalid_responses = {"there", "unknown", "none", "n/a", "na", "", "friend", "hey", "hi", "hello", "say", "yes", "no", "yeah", "okay", "ok"}
                if name and name.lower() not in invalid_responses and len(name) < 20 and ' ' not in name:
                    logger.debug(f"Name decoded by LLM: '{name}' from '{speech_text}'")
                    return name
                
        except Exception as e:
            logger.warning(f"LLM name extraction failed: {e}")
    
    # Fallback: take first word that looks like a name
    words = text.split()
    skip_words = {"hey", "hi", "hello", "yes", "no", "yeah", "um", "uh", "well", "so", 
                  "okay", "ok", "good", "fine", "great", "thanks", "thank", "you", 
                  "the", "a", "an", "is", "am", "are", "my", "name", "i", "i'm", "it's", "this"}
    
    for word in words:
        clean_word = word.strip(".,!?'\"")
        if clean_word.lower() not in skip_words and len(clean_word) >= 2 and clean_word.isalpha():
            name = clean_word.title()
            logger.debug(f"Name extracted from first valid word: '{name}' from '{speech_text}'")
            return name
    
    # Last resort: use the whole cleaned text as the name
    if text and len(text) < 20:
        name = text.split()[0].title() if text.split() else "Friend"
        logger.debug(f"Using first word as name: '{name}' from '{speech_text}'")
        return name
    
    return "Friend"


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
        # Fallback: keyword matching — default to NOT resolved unless explicitly positive
        text_lower = speech_text.lower()
        
        # Check RESOLVED phrases first — these are specific and take priority
        resolved_phrases = [
            "fixed", "that worked", "it worked", "working now", "all good",
            "problem solved", "resolved", "that helped", "it's working",
        ]
        if any(phrase in text_lower for phrase in resolved_phrases):
            return {"is_resolved": True, "confidence": "medium", "interpretation": speech_text}
        
        # Then check negative patterns — broader, so checked second
        negative_phrases = [
            "not working", "same issue", "didn't help", "didn't work", "worse",
            "no change", "nothing changed", "same problem", "still broken",
            "doesn't work", "doesn't help", "still not", "won't work",
            "no luck", "not fixed",
        ]
        negative_words = ["no", "nope", "didn't", "doesn't", "checked", "tried", "already"]
        
        if any(phrase in text_lower for phrase in negative_phrases):
            return {"is_resolved": False, "confidence": "medium", "interpretation": speech_text}
        # For single words, check as whole words to avoid substring false matches
        words_set = set(text_lower.split())
        if any(w in words_set for w in negative_words):
            return {"is_resolved": False, "confidence": "medium", "interpretation": speech_text}
        
        return {"is_resolved": False, "confidence": "low", "interpretation": speech_text}
    
    try:
        prompt = (
            "You are helping interpret a customer's response during appliance troubleshooting.\n\n"
            f'Troubleshooting step given: "{troubleshooting_step}"\n\n'
            f'Customer\'s response: "{speech_text}"\n\n'
            "Determine if the appliance issue is RESOLVED or still PERSISTS.\n\n"
            "CRITICAL RULES — read carefully:\n"
            "- Default to is_resolved: false UNLESS the customer EXPLICITLY says the problem is fixed.\n"
            "- The customer must use clear resolution language like 'that fixed it', 'it's working now',\n"
            "  'problem solved', 'yes that helped', 'all good now' for is_resolved to be true.\n"
            "- If the customer just says they checked something or tried a step, that does NOT mean resolved.\n"
            "  Checking a step ≠ problem fixed. They are reporting what they found.\n"
            "- 'It didn't work', 'no', 'still the same', 'not working', 'no change', 'nope',\n"
            "  'didn't help', 'same problem' → is_resolved: false, confidence: high\n"
            "- 'I checked it', 'I tried that', 'I looked at it', 'it's already set correctly',\n"
            "  'the setting is fine', 'it's plugged in' → is_resolved: false, confidence: high\n"
            "  (These mean the customer tried the step but the problem STILL EXISTS)\n"
            "- 'I don't know', ambiguous, or unrelated → is_resolved: false, confidence: low\n"
            "- ONLY mark is_resolved: true when customer EXPLICITLY confirms the fix worked.\n\n"
            "Respond in JSON format:\n"
            '{\n'
            '  "is_resolved": true/false,\n'
            '  "confidence": "high/medium/low",\n'
            '  "interpretation": "brief explanation of what customer meant"\n'
            '}\n\n'
            "Examples:\n\n"
            'Input: "No, it didn\'t work"\n'
            'Output: {"is_resolved": false, "confidence": "high", "interpretation": "Customer says troubleshooting did not help"}\n\n'
            'Input: "I checked the dial, it\'s already at max cooling"\n'
            'Output: {"is_resolved": false, "confidence": "high", "interpretation": "Setting was already correct, issue persists"}\n\n'
            'Input: "I tried all three steps but nothing changed"\n'
            'Output: {"is_resolved": false, "confidence": "high", "interpretation": "All steps tried, issue persists"}\n\n'
            'Input: "Nope, still not cooling"\n'
            'Output: {"is_resolved": false, "confidence": "high", "interpretation": "Issue persists after troubleshooting"}\n\n'
            'Input: "The door seems fine"\n'
            'Output: {"is_resolved": false, "confidence": "high", "interpretation": "Door is OK but overall issue persists"}\n\n'
            'Input: "Yes, that fixed it! It\'s working now!"\n'
            'Output: {"is_resolved": true, "confidence": "high", "interpretation": "Customer confirms issue is resolved"}\n\n'
            'Input: "Oh wow, it started working again!"\n'
            'Output: {"is_resolved": true, "confidence": "high", "interpretation": "Appliance is working after troubleshooting"}\n\n'
            "Now analyze:"
        )

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
        
        # Extract JSON object even if LLM included extra text around it
        json_match = re.search(r'\{[^{}]*\}', raw_result, re.DOTALL)
        if json_match:
            raw_result = json_match.group(0)
        
        if not raw_result or not raw_result.strip().startswith('{'):
            logger.warning(f"No JSON found in troubleshoot response: '{raw_result[:200]}', using fallback")
            raise ValueError("No JSON found")
        
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
1. Return ONLY the first name (e.g., "John", "Kasi", "Shiny")
2. If the input is noise, random words, or not a name, return "none"
3. Common patterns: "My name is John", "I'm Sarah", "This is Mike", or just "John"
4. Ignore filler words like "uh", "um", "whatever", "just", etc.
5. Accept names from ALL cultures and languages — Indian, Chinese, Arabic, African, etc.
6. A single word that could be a name IS a name. When in doubt, treat it as a name.

Examples:
- "My name is John Smith" -> John
- "I'm Sarah" -> Sarah
- "Shiny" -> Shiny
- "Kasi" -> Kasi
- "Priya" -> Priya
- "Hi Sam, my name is Wei" -> Wei
- "Whatever" -> none
- "Uh, just testing" -> none
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
    
    # Convert number words to digits — but ONLY when they appear in a numeric
    # context (e.g. "two four" in "majji two four").  In an email context,
    # sequences like "nine nine nine" are almost always the letter "n" repeated
    # (STT hears "n" as "nine").  Detect this: if 3+ consecutive number words
    # appear and there is NO @ yet AND we're in a letter-spelling context,
    # treat them as letters instead.
    words = text.split()
    converted = []
    i = 0
    while i < len(words):
        clean_word = re.sub(r'[.,;:!?]', '', words[i])
        if clean_word in _NUMBER_WORDS:
            # Look ahead: count consecutive number words
            run_start = i
            run = []
            while i < len(words):
                cw = re.sub(r'[.,;:!?]', '', words[i])
                if cw in _NUMBER_WORDS:
                    run.append((words[i], cw))
                    i += 1
                else:
                    break
            # Heuristic: if 3+ consecutive "nine" (or same number word),
            # it's likely the letter being spelled, not actual digits.
            # "nine nine nine" → "nnn" not "999"
            # Map: nine→n, eight→a (ate), five→f, four→f, six→s, two→t
            _DIGIT_TO_LETTER_GUESS = {
                'nine': 'n', 'niner': 'n',
                'eight': 'a', 'ate': 'a',
                'five': 'f',
                'four': 'f', 'for': 'f', 'fore': 'f',
                'six': 's', 'sicks': 's',
                'two': 't', 'to': 't', 'too': 't',
                'one': 'w', 'won': 'w',
            }
            unique_words = set(cw for _, cw in run)
            if len(run) >= 3 and len(unique_words) == 1 and list(unique_words)[0] in _DIGIT_TO_LETTER_GUESS:
                # All same number word repeated 3+ times → likely a letter
                letter = _DIGIT_TO_LETTER_GUESS[list(unique_words)[0]]
                converted.append(letter * len(run))
            else:
                # Normal digit conversion
                for orig, cw in run:
                    converted.append(_NUMBER_WORDS.get(cw, orig))
        else:
            converted.append(words[i])
            i += 1
    text = ' '.join(converted)
    
    # Collapse spaced single digits: "1 2 3" -> "123"
    text = re.sub(r'(\d)\s+(?=\d)', r'\1', text)
    
    # Collapse digits adjacent to letters with spaces: "majji 24" -> "majji24"
    # This handles non-native speakers who pause between letter groups and numbers
    text = re.sub(r'([a-z])\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+([a-z])', r'\1\2', text)
    
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


def llm_extract_email(speech_text: str) -> str:
    """
    Extract email address from Twilio speech-to-text output - NEVER returns None.
    
    Uses deterministic processing first, then LLM fallback to construct email.
    Always returns an email, even if it has to be constructed from the speech.
    
    Returns:
        Extracted or constructed email string. Never returns None.
    """
    if not speech_text or not speech_text.strip():
        logger.debug("Email extract: Empty input")
        return "customer@email.com"
    
    # Step 1: Deterministic pre-processing (handles all STT artifacts)
    normalized = _normalize_speech_for_email(speech_text)
    logger.debug(f"Email normalized: '{normalized}' from '{speech_text}'")
    
    # Step 2: Build email from normalized text
    email_candidate = re.sub(r'\s*@\s*', '@', normalized)
    email_candidate = email_candidate.rstrip('.,;:!?')
    
    logger.debug(f"Email candidate after cleanup: '{email_candidate}'")
    
    # Step 3: Extract email using regex
    # Always try the no-spaces version too, and prefer the one with the longer username
    match = _EMAIL_REGEX.search(email_candidate)
    email_no_spaces = email_candidate.replace(' ', '')
    match_no_spaces = _EMAIL_REGEX.search(email_no_spaces)
    
    if match_no_spaces:
        logger.debug(f"Email found after removing spaces: '{email_no_spaces}'")
    
    # Pick the best match: prefer the one with the longer username (more complete)
    best_match = None
    if match and match_no_spaces:
        user_orig = match.group(0).split('@')[0]
        user_nospace = match_no_spaces.group(0).split('@')[0]
        best_match = match_no_spaces if len(user_nospace) > len(user_orig) else match
    elif match_no_spaces:
        best_match = match_no_spaces
    elif match:
        best_match = match
    
    if best_match:
        email = best_match.group(0).lower()
        has_valid_tld = any(email.endswith(tld) for tld in _VALID_TLDS)
        if has_valid_tld:
            logger.info(f"Email extracted: '{email}'")
            return email
    
    # Step 4: LLM fallback - construct email from speech
    if model:
        try:
            prompt = f"""A customer spelled out their email address on a phone call.
The speech-to-text captured: "{speech_text}"

Decode and construct the complete email address. Consider:
- Letters may be spelled out with pauses: "k a s i" = "kasi"
- Periods between letters are STT artifacts, not real periods: "K. A. S. I." = "kasi"
- "at" or "at the rate" = "@"
- "dot" between name parts = "." (e.g. "kasi dot majji" = "kasi.majji")
- "dot com" = ".com", "dot net" = ".net"
- Numbers may be spoken: "two four" or "2 4" = "24"
- The speaker may be a non-native English speaker, so letters may sound different
- Common domains: gmail.com, yahoo.com, outlook.com, hotmail.com
- If no domain mentioned, assume @gmail.com
- Join ALL spelled letters and numbers into one continuous username before the @

Return ONLY the complete email address, nothing else:"""

            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 50}
            )
            
            try:
                raw_result = response.text.strip()
            except:
                if response.candidates and response.candidates[0].content.parts:
                    raw_result = response.candidates[0].content.parts[0].text.strip()
                else:
                    raw_result = ""
            
            email = raw_result.strip('"\'.,!? ').lower()
            
            # Validate it looks like an email
            if '@' in email and '.' in email.split('@')[-1]:
                logger.info(f"Email constructed by LLM: '{email}'")
                return email
                
        except Exception as e:
            logger.warning(f"LLM email extraction failed: {e}")
    
    # Step 5: Last resort - construct email from normalized text
    # Remove all spaces and non-alphanumeric chars except @ and .
    clean_text = re.sub(r'[^a-zA-Z0-9@.]', '', email_candidate.replace(' ', ''))
    
    if '@' in clean_text:
        # Has @ sign, try to fix it
        parts = clean_text.split('@')
        username = parts[0] if parts[0] else "customer"
        domain = parts[1] if len(parts) > 1 and parts[1] else "gmail.com"
        if '.' not in domain:
            domain = domain + ".com" if domain else "gmail.com"
        email = f"{username}@{domain}"
        logger.info(f"Email constructed from parts: '{email}'")
        return email
    else:
        # No @ sign - use the text as username with gmail.com
        username = clean_text if clean_text else "customer"
        email = f"{username}@gmail.com"
        logger.info(f"Email constructed with default domain: '{email}'")
        return email


def llm_analyze_customer_intent(speech_text: str) -> dict:
    """
    Analyze the customer's open-ended response to understand their intent.
    This powers the autonomous flow — the customer can say anything and the
    agent adapts: describe a problem, ask to schedule, mention an appliance, etc.
    
    Returns:
        dict with:
        - intent: 'describe_problem' | 'schedule_technician' | 'general_inquiry' | 'unclear'
        - appliance_type: str or None (washer, dryer, refrigerator, etc.)
        - symptoms: str or None (problem description if provided)
        - wants_scheduling: bool (True if customer explicitly wants to schedule)
        - has_full_description: bool (True if customer gave enough detail to skip symptom asking)
    """
    fallback = {
        "intent": "unclear",
        "appliance_type": None,
        "symptoms": None,
        "wants_scheduling": False,
        "has_full_description": False
    }
    
    if not speech_text or not speech_text.strip():
        return fallback
    
    # Quick keyword check for scheduling intent
    text_lower = speech_text.lower()
    scheduling_keywords = ["schedule", "technician", "appointment", "book", "visit", "come out", "send someone"]
    wants_scheduling = any(kw in text_lower for kw in scheduling_keywords)
    
    if not model:
        # Fallback: keyword-based analysis
        # Check compound words first to avoid substring false matches
        # e.g. "dishwasher" contains "washer", "air conditioner" contains "air"
        appliance = None
        if "dishwasher" in text_lower:
            appliance = "dishwasher"
        elif "air conditioner" in text_lower or "heat pump" in text_lower:
            appliance = "hvac"
        else:
            for kw in APPLIANCE_KEYWORDS:
                if kw in text_lower:
                    if kw in ("washer", "washing"):
                        appliance = "washer"
                    elif kw in ("dryer", "drying"):
                        appliance = "dryer"
                    elif kw in ("fridge", "refrigerator", "freezer"):
                        appliance = "refrigerator"
                    elif kw in ("oven", "stove", "range", "cooktop"):
                        appliance = "oven"
                    elif kw in ("hvac", "heating", "cooling", "ac", "furnace"):
                        appliance = "hvac"
                    break
        
        # Check if customer described a symptom (not just named an appliance)
        symptom_keywords = [
            "not cooling", "not working", "won't start", "leaking", "broken",
            "noise", "loud", "error", "won't turn", "not heating", "not spinning",
            "not draining", "won't drain", "smells", "smoking", "sparking",
            "vibrating", "shaking", "flooding", "overflowing", "beeping",
            "flashing", "frozen", "ice", "warm", "hot", "cold",
        ]
        has_symptom = any(kw in text_lower for kw in symptom_keywords)
        has_full = appliance is not None and has_symptom
        return {
            "intent": "schedule_technician" if wants_scheduling else ("describe_problem" if appliance else "unclear"),
            "appliance_type": appliance,
            "symptoms": speech_text if has_symptom else None,
            "wants_scheduling": wants_scheduling,
            "has_full_description": has_full
        }
    
    try:
        prompt = (
            "You are a customer service AI for a home appliance repair company.\n"
            "Analyze the customer's message and extract their intent.\n\n"
            f'Customer said: "{speech_text}"\n\n'
            "Determine:\n"
            '1. intent: What does the customer want?\n'
            '   - "describe_problem" if they\'re describing an appliance issue\n'
            '   - "schedule_technician" if they explicitly want to schedule/book a technician\n'
            '   - "general_inquiry" if asking a question\n'
            '   - "unclear" if you can\'t determine\n'
            "2. appliance_type: Which appliance? One of: washer, dryer, refrigerator, dishwasher, oven, hvac, or null\n"
            "3. symptoms: A brief summary of the problem they described, or null if none\n"
            "4. wants_scheduling: true if they mentioned wanting to schedule/book a technician\n"
            "5. has_full_description: true if the customer mentioned BOTH an appliance AND any problem/symptom.\n"
            "   IMPORTANT: Even a short description counts as full if it has an appliance + a symptom.\n"
            "   Examples of has_full_description = TRUE:\n"
            '     - "My refrigerator is not cooling" (appliance=refrigerator, symptom=not cooling)\n'
            '     - "Washer is leaking" (appliance=washer, symptom=leaking)\n'
            '     - "Dryer won\'t start" (appliance=dryer, symptom=won\'t start)\n'
            '     - "Dishwasher making loud noise" (appliance=dishwasher, symptom=loud noise)\n'
            "   Examples of has_full_description = FALSE:\n"
            '     - "I have a problem with my fridge" (appliance=refrigerator, but NO specific symptom)\n'
            '     - "Something is wrong" (no appliance, no symptom)\n'
            '     - "My washer" (appliance only, no symptom)\n\n'
            "Respond in JSON only:\n"
            '{\n'
            '  "intent": "...",\n'
            '  "appliance_type": "..." or null,\n'
            '  "symptoms": "..." or null,\n'
            '  "wants_scheduling": true/false,\n'
            '  "has_full_description": true/false\n'
            '}'
        )

        raw_result = ""
        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 256}
        )
        try:
            raw_result = result.text.strip()
        except (ValueError, AttributeError):
            # Log why the response failed
            if hasattr(result, 'candidates') and result.candidates:
                candidate = result.candidates[0]
                logger.warning(f"Intent LLM candidate finish_reason: {getattr(candidate, 'finish_reason', 'unknown')}")
                if hasattr(candidate, 'safety_ratings'):
                    logger.warning(f"Intent LLM safety_ratings: {candidate.safety_ratings}")
                try:
                    raw_result = candidate.content.parts[0].text.strip()
                except (AttributeError, IndexError) as inner_e:
                    logger.error(f"Cannot extract intent from candidates: {inner_e}")
                    pass  # fall through to keyword fallback
            elif hasattr(result, 'prompt_feedback'):
                logger.warning(f"Intent LLM prompt_feedback: {result.prompt_feedback}")
        
        logger.debug(f"Intent analysis raw LLM response: '{raw_result[:500]}'")
        
        if not raw_result:
            raise ValueError("Empty LLM response")
        
        # Clean JSON — strip markdown fences
        if '```json' in raw_result:
            raw_result = re.sub(r'```json\n?', '', raw_result)
            raw_result = raw_result.replace('```', '').strip()
        elif '```' in raw_result:
            raw_result = re.sub(r'```[a-z]*\n?', '', raw_result)
            raw_result = raw_result.replace('```', '').strip()
        
        # Try to extract JSON object if there's extra text around it
        json_match = re.search(r'\{[^{}]*\}', raw_result, re.DOTALL)
        if json_match:
            raw_result = json_match.group(0)
        
        parsed = json.loads(raw_result)
        
        # Validate appliance_type
        appliance = parsed.get("appliance_type")
        if appliance and appliance not in VALID_APPLIANCES:
            appliance = None
        if appliance == "other":
            appliance = None
        
        result_dict = {
            "intent": parsed.get("intent", "unclear"),
            "appliance_type": appliance,
            "symptoms": parsed.get("symptoms"),
            "wants_scheduling": bool(parsed.get("wants_scheduling", False)),
            "has_full_description": bool(parsed.get("has_full_description", False))
        }
        logger.debug(f"Intent analysis parsed: '{speech_text[:60]}' -> {result_dict}")
        return result_dict
        
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        logger.error(f"Intent analysis raw text was: '{raw_result[:300] if raw_result else 'EMPTY'}'")
        
        # Robust keyword fallback when LLM JSON parsing fails
        text_lower = speech_text.lower()
        kw_appliance = None
        if "dishwasher" in text_lower:
            kw_appliance = "dishwasher"
        elif "air conditioner" in text_lower:
            kw_appliance = "hvac"
        else:
            for kw in APPLIANCE_KEYWORDS:
                if kw in text_lower:
                    if kw in ("washer", "washing"):
                        kw_appliance = "washer"
                    elif kw in ("dryer", "drying"):
                        kw_appliance = "dryer"
                    elif kw in ("fridge", "refrigerator", "freezer"):
                        kw_appliance = "refrigerator"
                    elif kw in ("oven", "stove", "range", "cooktop"):
                        kw_appliance = "oven"
                    elif kw in ("hvac", "heating", "cooling", "ac", "furnace"):
                        kw_appliance = "hvac"
                    break
        
        kw_scheduling = any(kw in text_lower for kw in ["schedule", "technician", "appointment", "book", "visit", "come out", "send someone"])
        kw_has_detail = len(speech_text.split()) > 8
        
        kw_result = {
            "intent": "schedule_technician" if kw_scheduling else ("describe_problem" if kw_appliance else "unclear"),
            "appliance_type": kw_appliance,
            "symptoms": speech_text if kw_has_detail else None,
            "wants_scheduling": kw_scheduling,
            "has_full_description": kw_has_detail and kw_appliance is not None
        }
        logger.info(f"Intent keyword fallback: '{speech_text[:60]}' -> {kw_result}")
        return kw_result


def llm_plan_next_step(user_text: str, state: dict) -> str:
    """
    Goal-grounded autonomous planner.

    Returns the next executable step from a strict allowlist. This keeps the
    agent autonomous while preventing off-policy behavior.

    Cross-cutting exit detection is now LLM-powered instead of keyword-based.
    """
    current_step = state.get("step") or "greet_ask_name"
    text = (user_text or "").strip()

    # Cross-cutting exit: use LLM to detect "call back later" / goodbye intent
    # Only check if there's actual speech and we're not already at a terminal step
    # Skip exit detection for steps where "no" or "call back" is a valid in-step response
    skip_exit_steps = {
        "confirm_email", "confirm_zip", "confirm_resolution",
        "choose_slot", "greet_ask_name",
    }
    if text and current_step != "done" and current_step not in skip_exit_steps and model:
        try:
            prompt = f"""Is the caller CLEARLY trying to end the entire phone call?

Current conversation step: {current_step}
Caller said: "{text}"

Rules:
- "yes" ONLY if the caller's PRIMARY intent is to end the call entirely
  (e.g., "goodbye", "I'll call back another time", "I don't need help anymore", "hang up")
- "no" if they are answering a question, describing a problem, making a choice,
  saying "no" to a specific question, or continuing the conversation in any way
- "no" if they say "no" followed by something else (they're responding to a question)
- "no" if they mention scheduling, troubleshooting, photos, or any service topic
- When in doubt, ALWAYS say "no" — let the step handler deal with it

Return ONLY "yes" or "no":"""
            result = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 5},
            )
            answer = result.text.strip().lower()
            if answer.startswith("yes"):
                return "done"
        except Exception as e:
            logger.debug(f"Planner exit-detection failed: {e}")

    # Deterministic state guards for in-flight operations
    if state.get("appointment_booked") or state.get("resolved"):
        return "done"
    if state.get("pending_email"):
        return "confirm_email"
    if state.get("waiting_for_upload") and current_step not in ("speak_analysis", "after_analysis"):
        return "waiting_for_upload"
    if not state.get("customer_name"):
        return "greet_ask_name"

    # All conversation steps have their own built-in LLM routing / intent
    # analysis.  The planner must NOT bypass them — just return the current
    # step so each handler's own logic runs.
    return current_step


def llm_generate_troubleshooting_steps(appliance_type: str, symptom_summary: str = "") -> str:
    """
    Use LLM to generate appliance-specific troubleshooting steps instead of
    relying on a static lookup table.  Returns a concise numbered list suitable
    for reading aloud over the phone (3 steps max).
    """
    if not model:
        return ""

    symptom_ctx = f' The reported issue is: "{symptom_summary}".' if symptom_summary else ""
    try:
        prompt = (
            f"You are a home appliance repair expert. Generate exactly 3 quick "
            f"troubleshooting steps a customer can try RIGHT NOW for their "
            f"{appliance_type}.{symptom_ctx}\n\n"
            "Rules:\n"
            "- Each step must be a single clear sentence the customer can act on immediately\n"
            "- Use simple language suitable for reading aloud on a phone call\n"
            "- Focus on the most common fixes for this appliance and symptom\n"
            "- Do NOT include safety warnings or disclaimers\n"
            "- Format: Step 1: ... Step 2: ... Step 3: ...\n\n"
            "Steps:"
        )
        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 200},
        )
        raw = result.text.strip()
        # Ensure it starts with "Step 1"
        if "Step 1" not in raw:
            return ""
        return raw
    except Exception as e:
        logger.error(f"Troubleshooting generation failed: {e}")
        return ""


def llm_classify_yes_no(user_text: str, context: str = "") -> dict:
    """
    Universal LLM-powered yes/no/correction classifier.
    Replaces ALL keyword-based is_yes_response / is_no_response checks.

    Returns:
        {
          "intent": "yes" | "no" | "correction" | "unclear",
          "correction_value": str | None
        }
    """
    fallback = {"intent": "unclear", "correction_value": None}
    if not user_text or not user_text.strip():
        return fallback

    # Lightweight keyword fallback when LLM model is unavailable (tests, no API key)
    if not model:
        text_lower = user_text.lower().strip()
        # Check negatives FIRST — "incorrect" contains "correct" so order matters
        _no = {"no", "nope", "wrong", "incorrect", "negative", "not right",
               "that's wrong", "that is wrong", "try again"}
        _yes = {"yes", "yeah", "yep", "yup", "correct", "right", "sure", "ok", "okay",
                "affirmative", "absolutely", "that's right", "that's correct", "that is right"}
        if any(w in text_lower for w in _no):
            return {"intent": "no", "correction_value": None}
        if any(w in text_lower for w in _yes):
            return {"intent": "yes", "correction_value": None}
        return fallback

    try:
        prompt = f"""Classify the caller's response as yes, no, correction, or unclear.

Context: {context if context else "Agent asked a yes/no confirmation question."}
Caller said: "{user_text}"

Rules:
- "yes" = any affirmative (yes, yeah, yep, correct, that's right, sure, ok, absolutely, uh-huh, mm-hmm)
- "no" = any negative (no, nope, wrong, incorrect, that's wrong, negative, not right)
- "correction" = caller provides a corrected value (e.g. "no it's 60604", "actually it's john@gmail.com")
  Extract the corrected value into correction_value.
- "unclear" = cannot determine intent

Return ONLY valid JSON:
{{"intent": "...", "correction_value": null}}"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64},
        )
        raw = result.text.strip()
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        data = json.loads(raw)
        intent = data.get("intent", "unclear")
        if intent not in ("yes", "no", "correction", "unclear"):
            intent = "unclear"
        logger.debug(f"LLM yes/no: '{user_text}' -> {intent}")
        return {"intent": intent, "correction_value": data.get("correction_value")}
    except Exception as e:
        logger.warning(f"LLM yes/no failed: {e}")
        return fallback


def llm_classify_user_intent(user_text: str, choices: list[str], context: str = "") -> dict:
    """
    Universal LLM-powered multi-choice intent classifier.
    Replaces ALL keyword-list matching for routing decisions.

    Args:
        user_text: What the caller said
        choices: Valid choice labels, e.g. ["troubleshoot", "schedule", "callback", "photo"]
        context: What the agent just asked

    Returns:
        {"choice": one of choices or "unclear", "confidence": float 0-1}
    """
    fallback = {"choice": "unclear", "confidence": 0.0}
    if not user_text or not user_text.strip():
        return fallback

    # Lightweight keyword fallback when LLM model is unavailable (tests, no API key)
    if not model:
        text_lower = user_text.lower()
        # Generic keyword hints per common choice labels
        _choice_hints = {
            "troubleshoot": ["troubleshoot", "try", "steps", "fix myself", "diagnose"],
            "schedule": ["schedule", "technician", "appointment", "book", "visit", "send someone"],
            "callback": ["call back", "later", "not now", "another time", "goodbye"],
            "photo": ["photo", "picture", "image", "upload"],
            "resolved": ["fixed", "worked", "helped", "resolved", "all good"],
            "not_resolved": ["not working", "didn't help", "still broken", "same issue"],
            "cancel": ["cancel", "never mind", "hang up", "goodbye"],
            "select_slot": ["option", "first", "second", "third", "one", "two", "three"],
            "done": ["done", "uploaded", "finished", "sent"],
            "skip": ["skip", "schedule", "technician", "forget"],
            "more_time": ["wait", "more time", "minute", "hold on", "not yet"],
            "resend": ["resend", "send again", "another email", "didn't get"],
            "describe_problem": ["not working", "broken", "issue", "problem", "error"],
            "unsure": ["don't know", "not sure", "no idea"],
        }
        for c in choices:
            hints = _choice_hints.get(c, [])
            if any(h in text_lower for h in hints):
                return {"choice": c, "confidence": 0.75}
        return fallback

    try:
        choices_str = ", ".join(f'"{c}"' for c in choices)
        prompt = f"""Classify the caller's intent from their response.

Context: {context}
Valid choices: [{choices_str}]
Caller said: "{user_text}"

Rules:
- Pick the single best matching choice
- If the caller clearly wants one option, confidence should be >= 0.8
- If ambiguous, set confidence < 0.5
- If completely unrelated, use "unclear"

Return ONLY valid JSON:
{{"choice": "...", "confidence": 0.0}}"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64},
        )
        raw = result.text.strip()
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        data = json.loads(raw)
        choice = data.get("choice", "unclear")
        if choice not in choices and choice != "unclear":
            choice = "unclear"
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        logger.debug(f"LLM intent: '{user_text}' -> {choice} ({confidence:.2f})")
        return {"choice": choice, "confidence": confidence}
    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")
        return fallback


def llm_extract_zip_code(speech_text: str) -> str | None:
    """
    Use LLM to extract a 5-digit US ZIP code from natural speech.
    Handles spoken digits, number words, and STT artifacts.

    Returns 5-digit string or None.
    """
    if not speech_text or not speech_text.strip():
        return None

    # Quick regex check first — if 5+ digits exist, extract them
    digits = re.sub(r'\D', '', speech_text)
    if len(digits) >= 5:
        return digits[:5]

    if not model:
        return None

    try:
        prompt = f"""Extract the 5-digit US ZIP code from this phone call speech.

Speech: "{speech_text}"

Rules:
- Convert number words to digits: "six oh six oh one" = "60601"
- Handle STT artifacts: "6. 0. 6. 0. 1." = "60601"
- "triple nine" patterns: "nine nine nine" = "999"
- Return ONLY the 5 digits, nothing else
- If no valid ZIP code can be extracted, return "none"

ZIP code:"""
        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        raw = result.text.strip().replace(" ", "")
        clean = re.sub(r'\D', '', raw)
        if len(clean) == 5:
            return clean
        return None
    except Exception as e:
        logger.warning(f"LLM ZIP extraction failed: {e}")
        return None


def llm_extract_time_preference(speech_text: str) -> str | None:
    """
    Use LLM to extract morning/afternoon preference from natural speech.
    Returns "morning", "afternoon", or None.
    """
    if not speech_text or not speech_text.strip():
        return None
    if not model:
        text_lower = speech_text.lower()
        if "morning" in text_lower:
            return "morning"
        if "afternoon" in text_lower or "evening" in text_lower:
            return "afternoon"
        return None

    try:
        prompt = f"""The caller was asked if they prefer a morning or afternoon appointment.

Caller said: "{speech_text}"

Rules:
- "morning", "early", "AM", "before noon" → morning
- "afternoon", "evening", "PM", "after lunch", "later in the day" → afternoon
- "anytime", "doesn't matter", "either" → anytime
- If unclear → unclear

Return ONLY one word: morning, afternoon, anytime, or unclear"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        raw = result.text.strip().lower()
        if raw in ("morning", "afternoon", "anytime"):
            return raw if raw != "anytime" else None
        return None
    except Exception as e:
        logger.warning(f"LLM time pref extraction failed: {e}")
        return None


def llm_choose_slot(speech_text: str, slots_description: str) -> int | None:
    """
    Use LLM to match the caller's slot selection to one of the offered slots.
    Returns 0-based index or None.
    """
    if not speech_text or not speech_text.strip():
        return None
    if not model:
        return None

    try:
        prompt = f"""The caller was offered appointment slots and needs to pick one.

Available slots:
{slots_description}

Caller said: "{speech_text}"

Rules:
- Match by option number: "option 1", "the first one", "one" → 0
- Match by day name: "Sunday", "Monday" → index of that slot
- Match by date: "February 15th" → index of that slot
- Match by time: "the morning one", "the 9 AM" → index of that slot
- Return ONLY the 0-based index number (0, 1, or 2)
- If cannot determine, return "none"

Index:"""
        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        raw = result.text.strip()
        clean = re.sub(r'\D', '', raw)
        if clean and int(clean) in (0, 1, 2):
            return int(clean)
        return None
    except Exception as e:
        logger.warning(f"LLM slot selection failed: {e}")
        return None


def llm_interpret_upload_intent(speech_text: str) -> str:
    """
    Interpret caller's intent during the upload waiting step.
    Returns one of: "done", "skip", "more_time", "resend", "unclear"
    """
    if not speech_text or not speech_text.strip():
        return "unclear"
    if not model:
        return "unclear"

    try:
        prompt = f"""The caller is on hold while uploading a photo via email link.

Caller said: "{speech_text}"

Classify their intent:
- "done" = they finished uploading (done, uploaded, finished, sent it, I did it)
- "skip" = they want to skip upload and schedule a technician (skip, schedule, technician, forget it, just book)
- "more_time" = they need more time (yes, wait, more time, one minute, hold on, not yet)
- "resend" = they want the email link resent (send again, resend, didn't get it, another email)
- "unclear" = cannot determine

Return ONLY one word: done, skip, more_time, resend, or unclear"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        raw = result.text.strip().lower().split()[0] if result.text.strip() else "unclear"
        if raw in ("done", "skip", "more_time", "resend"):
            return raw
        return "unclear"
    except Exception as e:
        logger.warning(f"LLM upload intent failed: {e}")
        return "unclear"


def llm_interpret_after_analysis(speech_text: str) -> str:
    """
    Interpret caller's response after hearing image analysis results.
    Returns one of: "resolved", "schedule", "try_fix", "unclear"
    """
    if not speech_text or not speech_text.strip():
        return "unclear"
    if not model:
        return "unclear"

    try:
        prompt = f"""The caller just heard AI analysis of their appliance photo with a suggested fix.
The agent asked: "Would you like to try that, or should I schedule a technician?"

Caller said: "{speech_text}"

Classify their intent:
- "resolved" = issue is fixed / they're satisfied (it worked, that fixed it, all good, yes it helped, great)
- "schedule" = they want a technician (schedule, technician, send someone, didn't work, still broken, no luck, not working)
- "try_fix" = they want to try the suggested fix (I'll try, let me try, okay I'll do that, sure)
- "unclear" = cannot determine

Return ONLY one word: resolved, schedule, try_fix, or unclear"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 10},
        )
        raw = result.text.strip().lower().split()[0] if result.text.strip() else "unclear"
        if raw in ("resolved", "schedule", "try_fix"):
            return raw
        return "unclear"
    except Exception as e:
        logger.warning(f"LLM after-analysis intent failed: {e}")
        return "unclear"


def llm_classify_confirmation(user_text: str, context: str = "") -> dict:
    """
    LLM fallback for confirmation intent when keywords don't match.
    
    Returns dict with:
    - intent: "yes" | "no" | "correction" | "unclear"
    - correction_value: str (if intent is "correction", the new value)
    
    Example inputs:
    - "that's right" → {"intent": "yes"}
    - "correct" → {"intent": "yes"}
    - "no it's 60604" → {"intent": "correction", "correction_value": "60604"}
    - "actually 60604" → {"intent": "correction", "correction_value": "60604"}
    - "hmm let me think" → {"intent": "unclear"}
    """
    fallback = {"intent": "unclear", "correction_value": None}
    
    if not model:
        return fallback
    
    try:
        prompt = f"""Classify the user's response to a confirmation question.

Context: {context if context else "Agent asked for yes/no confirmation"}
User said: "{user_text}"

Return ONLY valid JSON with:
- "intent": one of "yes", "no", "correction", "unclear"
- "correction_value": if intent is "correction", extract the corrected value (e.g., ZIP code, email); otherwise null

Examples:
- "that's right" → {{"intent": "yes", "correction_value": null}}
- "correct" → {{"intent": "yes", "correction_value": null}}
- "yep" → {{"intent": "yes", "correction_value": null}}
- "no" → {{"intent": "no", "correction_value": null}}
- "no it's 60604" → {{"intent": "correction", "correction_value": "60604"}}
- "actually 60604" → {{"intent": "correction", "correction_value": "60604"}}
- "wait let me check" → {{"intent": "unclear", "correction_value": null}}

JSON:"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64}
        )
        raw = result.text.strip()
        
        # Extract JSON from response
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        data = json.loads(raw)
        intent = data.get("intent", "unclear")
        if intent not in ("yes", "no", "correction", "unclear"):
            intent = "unclear"
        
        logger.debug(f"LLM confirmation: '{user_text}' -> {data}")
        return {
            "intent": intent,
            "correction_value": data.get("correction_value")
        }
        
    except Exception as e:
        logger.warning(f"LLM confirmation fallback failed: {e}")
        return fallback


def llm_classify_choice(user_text: str, choices: list[str], context: str = "") -> dict:
    """
    LLM fallback for choice/routing intent when keywords don't match.
    
    Args:
        user_text: What the user said
        choices: List of valid choices, e.g. ["troubleshoot", "schedule", "callback", "photo"]
        context: Optional context about what was offered
    
    Returns dict with:
    - choice: one of the provided choices, or "unclear"
    - confidence: float 0-1
    
    Example:
    - "I think I want to try fixing it myself" → {"choice": "troubleshoot", "confidence": 0.9}
    - "just send someone" → {"choice": "schedule", "confidence": 0.95}
    - "I'll call back another time" → {"choice": "callback", "confidence": 0.9}
    """
    fallback = {"choice": "unclear", "confidence": 0.0}
    
    if not model:
        return fallback
    
    try:
        choices_str = ", ".join(f'"{c}"' for c in choices)
        prompt = f"""Classify the user's choice from their response.

Context: {context if context else "Agent offered multiple options"}
Valid choices: [{choices_str}]
User said: "{user_text}"

Return ONLY valid JSON with:
- "choice": one of [{choices_str}] or "unclear" if can't determine
- "confidence": float between 0 and 1

Examples for choices ["troubleshoot", "schedule", "callback"]:
- "I want to try fixing it" → {{"choice": "troubleshoot", "confidence": 0.9}}
- "let's try the steps" → {{"choice": "troubleshoot", "confidence": 0.85}}
- "just send a technician" → {{"choice": "schedule", "confidence": 0.95}}
- "I'll call back later" → {{"choice": "callback", "confidence": 0.9}}
- "hmm not sure" → {{"choice": "unclear", "confidence": 0.3}}

JSON:"""

        result = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 64}
        )
        raw = result.text.strip()
        
        # Extract JSON from response
        if "```" in raw:
            raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        data = json.loads(raw)
        choice = data.get("choice", "unclear")
        if choice not in choices and choice != "unclear":
            choice = "unclear"
        
        confidence = float(data.get("confidence", 0.0))
        
        logger.debug(f"LLM choice: '{user_text}' -> {choice} (conf={confidence:.2f})")
        return {"choice": choice, "confidence": confidence}
        
    except Exception as e:
        logger.warning(f"LLM choice fallback failed: {e}")
        return fallback


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
        prompt = f"""You are a friendly phone agent for a home appliance repair company.
The customer just described their appliance problem. Summarize it in a way you can
speak back to them naturally on the phone.

Always respond in valid JSON with exactly these keys:
- "symptom_summary": string — a SHORT natural sentence you will say back to the customer.
  MUST be written in 2nd person ("your refrigerator...", "it sounds like your washer...").
  NEVER use 3rd person like "The customer reported", "The caller described", "The user said".
  NEVER include meta-commentary like "no error codes mentioned" or "no specific symptoms".
  Keep it to ONE short sentence, max 15 words.
  Examples of GOOD summaries:
    "Your refrigerator isn't cooling properly"
    "Your washer is leaking water during the spin cycle"
    "Your dryer isn't heating up"
    "Your dishwasher won't start"
  Examples of BAD summaries (NEVER do this):
    "The customer reported that their refrigerator is not cooling"
    "Caller describes a leaking washer with no error codes"
    "The user's dryer is not heating. No error codes were mentioned."
- "error_codes": list of strings (error codes like "E23", "F21", etc. — empty list if none)
- "is_urgent": boolean (true ONLY if safety issue: flooding, fire risk, gas smell, sparking)

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
