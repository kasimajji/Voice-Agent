import re
from fastapi import APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse

from .config import APP_BASE_URL
from .conversation import (
    get_state,
    update_state,
    infer_appliance_type,
    get_next_troubleshooting_prompt,
    is_positive_response,
)
from .llm import llm_classify_appliance, llm_extract_symptoms, llm_is_appliance_related, llm_extract_email
from .scheduling import find_available_slots, book_appointment, format_slot_for_speech
from .image_service import (
    create_image_upload_token,
    build_upload_url,
    send_upload_email,
    validate_email,
    get_upload_status_by_call_sid
)

router = APIRouter()

# Absolute URL for Twilio webhooks
VOICE_CONTINUE_URL = f"{APP_BASE_URL}/twilio/voice/continue"

# Maximum polling attempts for image upload (Issue 2)
MAX_UPLOAD_POLL_COUNT = 10  # ~2.5 minutes with 15s pauses


def extract_email_from_speech(speech_text: str) -> str | None:
    """
    Extract email from Twilio speech-to-text using AI.
    
    Delegates to llm_extract_email for intelligent extraction that handles:
    - Spelled out letters: "k a s i at gmail dot com"
    - Twilio artifacts: "K. A s. I dot m. A j. J. I at gmail.com."
    - Common phrasings: "at the rate", "dot com"
    
    Returns:
        Extracted email string if found, None otherwise.
    """
    print(f"[Email Extract] Raw input: {speech_text}")
    email = llm_extract_email(speech_text)
    print(f"[Email Extract] LLM result: {email}")
    return email


def spell_email_for_speech(email: str) -> str:
    """
    Convert email to speakable format for voice confirmation.
    
    Example: "kasi.majji@gmail.com" → "k a s i dot m a j j i at g m a i l dot com"
    
    Handles:
    - @ → "at"
    - . → "dot"
    - _ → "underscore"
    - - → "dash"
    - Letters and numbers are spelled out with spaces
    - Skips any weird characters that shouldn't be spoken
    """
    if not email:
        return ""
    
    result = []
    for char in email.lower():
        if char == '@':
            result.append(" at ")
        elif char == '.':
            result.append(" dot ")
        elif char == '_':
            result.append(" underscore ")
        elif char == '-':
            result.append(" dash ")
        elif char == '+':
            result.append(" plus ")
        elif char.isalnum():
            result.append(f" {char} ")
        # Skip any other characters (don't speak them)
    
    # Join and clean up multiple spaces
    spelled = " ".join("".join(result).split())
    print(f"[Email Spell] {email} → {spelled}")
    return spelled


def is_yes_response(text: str) -> bool:
    """Check if user response is affirmative."""
    text_lower = text.lower().strip()
    yes_words = {"yes", "yeah", "yep", "yup", "correct", "right", "that's right", 
                 "that is right", "that's correct", "affirmative", "ok", "okay"}
    return any(word in text_lower for word in yes_words)


def is_no_response(text: str) -> bool:
    """Check if user response is negative."""
    text_lower = text.lower().strip()
    no_words = {"no", "nope", "wrong", "incorrect", "that's wrong", "not right",
                "that is wrong", "negative", "try again"}
    return any(word in text_lower for word in no_words)


@router.post("/voice")
async def voice_entry(request: Request):
    """Entry point when a call starts - Twilio hits this webhook."""
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid", "")
    from_number = form_data.get("From", "")
    to_number = form_data.get("To", "")
    
    print(f"[Incoming Call] CallSid: {call_sid}, From: {from_number}, To: {to_number}")
    
    state = get_state(call_sid)
    state["step"] = "greet_ask_name"
    state["customer_phone"] = from_number
    update_state(call_sid, state)
    
    response = VoiceResponse()
    
    # Natural greeting - warm and friendly
    gather = response.gather(
        input="speech",
        timeout=6,          # Give more time to respond
        speech_timeout="4", # Wait longer for natural pauses
        action=VOICE_CONTINUE_URL,
        method="POST"
    )
    gather.say(
        "Hi there! Thanks for calling Sears Home Services. "
        "My name is Sam, and I'll be helping you today. "
        "May I have your name, please?"
    )
    
    response.redirect(VOICE_CONTINUE_URL)
    
    twiml_string = str(response)
    return Response(content=twiml_string, media_type="application/xml")


@router.post("/voice/continue")
async def voice_continue(request: Request):
    """Handles the response after the user speaks - implements state machine."""
    call_sid = ""
    try:
        form_data = await request.form()
        
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")
        
        print(f"[Speech Received] CallSid: {call_sid}, SpeechResult: {speech_result}")
        
        state = get_state(call_sid)
        response = VoiceResponse()
        
        speech_result = speech_result or ""
        
        # Main state machine logic wrapped in inner try for graceful degradation
        return await _handle_voice_continue(call_sid, speech_result, state, response)
        
    except Exception as e:
        # Critical error handler - ensures call never crashes silently
        print(f"[CRITICAL ERROR] CallSid: {call_sid}, Error: {type(e).__name__}: {e}")
        response = VoiceResponse()
        response.say(
            "I'm sorry, we're experiencing technical difficulties. "
            "Please call back in a few minutes. Goodbye."
        )
        response.hangup()
        return Response(content=str(response), media_type="application/xml")


async def _handle_voice_continue(call_sid: str, speech_result: str, state: dict, response: VoiceResponse):
    """Inner handler for voice_continue - separated for cleaner error handling."""
    
    if not speech_result.strip():
        state["no_input_attempts"] = state.get("no_input_attempts", 0) + 1
        update_state(call_sid, state)
        
        if state["no_input_attempts"] <= 2:
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "I'm sorry, I didn't hear anything. "
                "Please say that again after the beep."
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            response.say(
                "I'm still not hearing anything clearly, "
                "so I'll end the call here. "
                "You can call us back any time. Goodbye."
            )
            response.hangup()
        
        return Response(content=str(response), media_type="application/xml")
    
    current_step = state.get("step", "greet_ask_name")
    
    # ==================== NATURAL CONVERSATION FLOW ====================
    
    if current_step == "greet_ask_name":
        # Extract name from speech (simple extraction - take the main content)
        customer_name = speech_result.strip()
        # Clean up common prefixes
        for prefix in ["my name is ", "i'm ", "this is ", "it's ", "i am ", "hey ", "hi "]:
            if customer_name.lower().startswith(prefix):
                customer_name = customer_name[len(prefix):]
                break
        # Take first word/name if multiple words
        customer_name = customer_name.split()[0] if customer_name.split() else "there"
        customer_name = customer_name.strip(".,!?").title()
        
        state["customer_name"] = customer_name
        state["step"] = "ask_how_are_you"
        update_state(call_sid, state)
        
        print(f"[Conversation] CallSid: {call_sid}, Customer name: {customer_name}")
        
        gather = response.gather(
            input="speech",
            timeout=6,
            speech_timeout="4",
            action=VOICE_CONTINUE_URL,
            method="POST"
        )
        gather.say(
            f"Nice to meet you, {customer_name}! How are you doing today?"
        )
        response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "ask_how_are_you":
        # Acknowledge their response warmly, then transition to help
        customer_name = state.get("customer_name", "there")
        state["step"] = "ask_appliance"
        update_state(call_sid, state)
        
        print(f"[Conversation] CallSid: {call_sid}, How are you response: {speech_result}")
        
        gather = response.gather(
            input="speech",
            timeout=10,
            speech_timeout="5",
            action=VOICE_CONTINUE_URL,
            method="POST"
        )
        gather.say(
            f"That's great to hear! "
            f"So {customer_name}, what can I help you with today? "
            "Are you having an issue with an appliance like a washer, dryer, refrigerator, dishwasher, oven, or HVAC system?"
        )
        response.redirect(VOICE_CONTINUE_URL)
    
    # ==================== APPLIANCE & TROUBLESHOOTING FLOW ====================
    
    elif current_step == "ask_appliance":
        # First validate if the input is appliance-related
        is_related = llm_is_appliance_related(speech_result)
        
        appliance = None
        if is_related:
            # Try Gemini LLM first, fall back to keyword-based inference
            appliance = llm_classify_appliance(speech_result)
            if not appliance:
                appliance = infer_appliance_type(speech_result)
                if appliance:
                    print(f"[Fallback] Using keyword-based appliance detection: {appliance}")
        else:
            print(f"[Validation] Input not appliance-related: '{speech_result}'")
        
        customer_name = state.get("customer_name", "")
        name_phrase = f", {customer_name}" if customer_name else ""
        
        if appliance:
            state["appliance_type"] = appliance
            state["step"] = "ask_symptoms"
            state["appliance_attempts"] = 0  # Reset on success
            update_state(call_sid, state)
            
            print(f"[State Update] CallSid: {call_sid}, Appliance: {appliance}")
            
            gather = response.gather(
                input="speech",
                timeout=10,
                speech_timeout="5",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"Got it{name_phrase}! So you're having trouble with your {appliance}. "
                f"Can you tell me a bit more about what's happening? "
                "For example, any error codes, strange noises, or specific issues you've noticed?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            state["appliance_attempts"] = state.get("appliance_attempts", 0) + 1
            update_state(call_sid, state)
            
            print(f"[Validation] Appliance attempt {state['appliance_attempts']}/2")
            
            if state["appliance_attempts"] < 2:
                # Retry - ask again naturally
                gather = response.gather(
                    input="speech",
                    timeout=10,
                    speech_timeout="5",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"I'm sorry{name_phrase}, I didn't quite catch that. "
                    "Which appliance is giving you trouble? "
                    "It could be a washer, dryer, fridge, dishwasher, oven, or your heating and cooling system."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # After 2 failed attempts, move to troubleshooting with unknown appliance
                state["appliance_type"] = "appliance"  # Generic fallback
                state["step"] = "ask_symptoms"
                update_state(call_sid, state)
                
                print(f"[Validation] Max attempts reached, proceeding with generic appliance")
                
                gather = response.gather(
                    input="speech",
                    timeout=10,
                    speech_timeout="5",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"No worries{name_phrase}! Let's figure this out together. "
                    "Can you describe what's going wrong? Tell me about the problem you're experiencing."
                )
                response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "ask_symptoms":
        # Store raw symptoms
        state["symptoms"] = speech_result
        customer_name = state.get("customer_name", "")
        name_phrase = f", {customer_name}" if customer_name else ""
        
        # Use Gemini to extract structured symptom information
        extracted = llm_extract_symptoms(speech_result)
        state["symptom_summary"] = extracted.get("symptom_summary") or speech_result
        state["error_codes"] = extracted.get("error_codes") or []
        state["is_urgent"] = bool(extracted.get("is_urgent"))
        
        state["step"] = "troubleshoot"
        state["troubleshooting_step"] = 0
        update_state(call_sid, state)
        
        print(f"[State Update] CallSid: {call_sid}, Symptoms: {speech_result}")
        print(f"[State Update] CallSid: {call_sid}, Summary: {state['symptom_summary']}, Errors: {state['error_codes']}, Urgent: {state['is_urgent']}")
        
        prompt = get_next_troubleshooting_prompt(state)
        update_state(call_sid, state)
        
        if prompt:
            gather = response.gather(
                input="speech",
                timeout=15,
                speech_timeout="5",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            # Use the LLM-generated symptom summary in the response
            summary = state["symptom_summary"]
            gather.say(
                f"Okay{name_phrase}, I understand. {summary}. "
                f"Let me help you with a quick troubleshooting step. {prompt} "
                "Take your time, and when you're ready, just say 'yes' if that helped or 'no' if you're still having the issue."
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            state["step"] = "done"
            update_state(call_sid, state)
            response.say(
                f"I'm sorry{name_phrase}, I don't have specific troubleshooting steps for that issue yet. "
                "But don't worry, our technicians can definitely help. "
                "Please call back to speak with a specialist. Thank you for calling Sears Home Services. Goodbye!"
            )
            response.hangup()
    
    elif current_step == "troubleshoot":
        customer_name = state.get("customer_name", "")
        name_phrase = f", {customer_name}" if customer_name else ""
        
        if is_positive_response(speech_result):
            state["resolved"] = True
            state["step"] = "done"
            update_state(call_sid, state)
            
            print(f"[Call Resolved] CallSid: {call_sid}")
            
            response.say(
                f"Wonderful{name_phrase}! I'm so glad that worked! "
                "If you ever have any other issues, don't hesitate to give us a call. "
                "Have a great day, and thank you for choosing Sears Home Services. Take care!"
            )
            response.hangup()
        else:
            prompt = get_next_troubleshooting_prompt(state)
            update_state(call_sid, state)
            
            if prompt:
                gather = response.gather(
                    input="speech",
                    timeout=15,
                    speech_timeout="5",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"Alright{name_phrase}, no problem. Let's try something else. {prompt} "
                    "Again, take your time and let me know if that helps."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Offer Tier 3 image upload option before scheduling
                state["step"] = "offer_image_upload"
                update_state(call_sid, state)
                
                print(f"[Escalation Needed] CallSid: {call_sid} - Offering Tier 3 image upload")
                
                gather = response.gather(
                    input="speech",
                    timeout=10,
                    speech_timeout="5",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"I understand{name_phrase}, sometimes these issues need a closer look. "
                    "I have a couple of options that might help. "
                    "I can send you a link to upload a photo of your appliance, "
                    "and our AI will analyze it and give you more specific advice. "
                    "Or, if you'd prefer, I can help you schedule a technician to come take a look in person. "
                    "What would you prefer - upload a photo, or schedule a technician visit?"
                )
                response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "offer_image_upload":
        text_lower = speech_result.lower()
        customer_name = state.get("customer_name", "")
        name_phrase = f", {customer_name}" if customer_name else ""
        
        if "photo" in text_lower or "picture" in text_lower or "image" in text_lower or "upload" in text_lower:
            # User wants Tier 3 image upload
            state["step"] = "collect_email"
            state["email_attempts"] = 0  # Reset email attempts
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - User chose image upload")
            
            # Enhanced Gather for initial email collection
            gather = response.gather(
                input="speech",
                timeout=15,
                speech_timeout="5",
                action=VOICE_CONTINUE_URL,
                method="POST",
                language="en-US",
                hints="gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                      "at gmail dot com, dot com, dot net, at the rate, "
                      "zero, one, two, three, four, five, six, seven, eight, nine, "
                      "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
            )
            gather.say(
                f"Perfect{name_phrase}! I'll send you a link to upload your photo. "
                "What's your email address? You can spell it out letter by letter if that's easier. "
                "For example: j, o, h, n, at, gmail, dot, com."
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        elif "technician" in text_lower or "schedule" in text_lower or "appointment" in text_lower or "visit" in text_lower:
            # User wants Tier 2 scheduling
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            print(f"[Tier 2] CallSid: {call_sid} - User chose technician scheduling")
            
            gather = response.gather(
                input="speech",
                timeout=10,
                speech_timeout="5",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"Absolutely{name_phrase}! Let me help you schedule a technician visit. "
                "To find available technicians in your area, could you please tell me your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Unclear response, ask again
            gather = response.gather(
                input="speech",
                timeout=10,
                speech_timeout="5",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "I'm sorry, I didn't catch that. "
                "Would you like to upload a photo for diagnosis, "
                "or schedule a technician visit?"
            )
            response.redirect(VOICE_CONTINUE_URL)
    
    # =========================================================================
    # ISSUE 1: Email capture with confirmation loop
    # States: collect_email → confirm_email → (email_confirmed or retry)
    # =========================================================================
    
    elif current_step == "collect_email":
        # ISSUE 1: Extract and validate email, then move to confirmation
        email = extract_email_from_speech(speech_result)
        
        if email and validate_email(email):
            # Store as pending - NOT confirmed yet
            state["pending_email"] = email
            state["step"] = "confirm_email"
            # Only initialize counter on first entry, preserve across retries
            if "email_confirm_attempts" not in state:
                state["email_confirm_attempts"] = 0
            update_state(call_sid, state)
            
            # Spell back the email for confirmation
            spelled_email = spell_email_for_speech(email)
            
            print(f"[Tier 3] CallSid: {call_sid} - Email captured: {email}, awaiting confirmation")
            
            # Confirmation gather - yes/no response
            gather = response.gather(
                input="speech",
                timeout=7,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST",
                language="en-US"
            )
            gather.say(
                f"I heard {spelled_email}. "
                "Is that correct? Please say yes or no."
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            state["email_attempts"] = state.get("email_attempts", 0) + 1
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - Email attempt {state['email_attempts']}, extracted: {email}")
            
            if state["email_attempts"] <= 3:
                # Retry email capture with enhanced Gather settings
                gather = response.gather(
                    input="speech",
                    timeout=10,  # Longer timeout for spelling
                    speech_timeout="4",  # More pause tolerance
                    action=VOICE_CONTINUE_URL,
                    method="POST",
                    language="en-US",
                    hints="gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                          "at gmail dot com, dot com, dot net, at the rate, "
                          "zero, one, two, three, four, five, six, seven, eight, nine, "
                          "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
                )
                gather.say(
                    "I'm sorry, I didn't catch a valid email address. "
                    "Please spell your email slowly, letter by letter. "
                    "For example: k, a, s, i, dot, m, a, j, j, i, at, gmail, dot, com."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Too many failed attempts, fall back to scheduling
                print(f"[Tier 3] CallSid: {call_sid} - Email capture failed after 3 attempts, falling back to scheduling")
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST",
                    language="en-US"
                )
                gather.say(
                    "I'm having trouble capturing the email. "
                    "Let me help you schedule a technician instead. "
                    "What is your ZIP code?"
                )
                response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "confirm_email":
        # ISSUE 1: User confirms or rejects the email
        pending_email = state.get("pending_email")
        
        if is_yes_response(speech_result):
            # Email confirmed! Now create token and send
            state["customer_email"] = pending_email
            state["pending_email"] = None
            
            print(f"[Tier 3] CallSid: {call_sid} - Email confirmed: {pending_email}")
            
            # Create upload token and send email
            try:
                upload_token = create_image_upload_token(
                    call_sid=call_sid,
                    email=pending_email,
                    appliance_type=state.get("appliance_type"),
                    symptom_summary=state.get("symptom_summary")
                )
                
                upload_url = build_upload_url(upload_token.token)
                send_upload_email(pending_email, upload_url, state.get("appliance_type"))
                
                state["image_upload_sent"] = True
                state["upload_token"] = upload_token.token
                state["waiting_for_upload"] = True
                state["upload_poll_count"] = 0
                state["step"] = "waiting_for_upload"
                update_state(call_sid, state)
                
                print(f"[Tier 3] CallSid: {call_sid} - Upload link sent, entering wait loop")
                
                # ISSUE 2: Keep call alive while waiting for upload
                gather = response.gather(
                    input="speech",
                    timeout=15,  # Longer timeout to give user time
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"I've sent an upload link to your email. "
                    "Please check your inbox, click the link, and upload a clear photo of your appliance. "
                    "I'll stay on the line while you do this. "
                    "Once you've uploaded the image, just say 'done' or 'uploaded'. "
                    "If you'd rather skip and schedule a technician, say 'skip'."
                )
                response.redirect(VOICE_CONTINUE_URL)
                
            except Exception as e:
                print(f"[Tier 3] CallSid: {call_sid} - Error creating token: {e}")
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I'm sorry, there was an issue sending the upload link. "
                    "Let me help you schedule a technician instead. "
                    "What is your ZIP code?"
                )
                response.redirect(VOICE_CONTINUE_URL)
        
        elif is_no_response(speech_result):
            # Email was wrong, retry
            state["email_confirm_attempts"] = state.get("email_confirm_attempts", 0) + 1
            state["pending_email"] = None
            
            print(f"[Tier 3] CallSid: {call_sid} - Email rejected, attempt {state['email_confirm_attempts']}")
            
            if state["email_confirm_attempts"] <= 2:
                state["step"] = "collect_email"
                update_state(call_sid, state)
                
                # Retry with enhanced Gather for email
                gather = response.gather(
                    input="speech",
                    timeout=10,
                    speech_timeout="4",
                    action=VOICE_CONTINUE_URL,
                    method="POST",
                    language="en-US",
                    hints="gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                          "at gmail dot com, dot com, dot net, at the rate, "
                          "zero, one, two, three, four, five, six, seven, eight, nine, "
                          "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
                )
                gather.say(
                    "No problem, let's try again. "
                    "Please spell your email slowly, letter by letter. "
                    "Say 'dot' for periods and 'at' for the at symbol."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Too many confirmation failures
                print(f"[Tier 3] CallSid: {call_sid} - Email confirmation failed 3 times, falling back")
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST",
                    language="en-US"
                )
                gather.say(
                    "I'm having trouble with the email. "
                    "Let me help you schedule a technician instead. "
                    "What is your ZIP code?"
                )
                response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Unclear response, ask again for yes/no
            gather = response.gather(
                input="speech",
                timeout=7,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST",
                language="en-US"
            )
            spelled_email = spell_email_for_speech(pending_email) if pending_email else "the email"
            gather.say(
                f"I need a yes or no. Is {spelled_email} correct?"
            )
            response.redirect(VOICE_CONTINUE_URL)
    
    # =========================================================================
    # ISSUE 2: Keep call alive during image upload, poll for completion
    # =========================================================================
    
    elif current_step == "waiting_for_upload":
        text_lower = speech_result.lower()
        
        # Check if user says they're done or wants to skip
        if "done" in text_lower or "uploaded" in text_lower or "finished" in text_lower:
            # User claims upload is done - check DB
            upload_status = get_upload_status_by_call_sid(call_sid)
            
            if upload_status and upload_status.get("analysis_ready"):
                # Analysis is ready - speak results
                state["step"] = "speak_analysis"
                update_state(call_sid, state)
                
                print(f"[Tier 3] CallSid: {call_sid} - Image uploaded and analyzed, speaking results")
                
                # Redirect to speak_analysis handling
                response.redirect(VOICE_CONTINUE_URL)
            
            elif upload_status and upload_status.get("image_uploaded"):
                # Image uploaded but analysis not ready yet
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                gather = response.gather(
                    input="speech",
                    timeout=10,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I see your image was received. Just a moment while I analyze it. "
                    "Say 'ready' when you'd like me to check again."
                )
                # Add a pause to give analysis time
                response.pause(length=5)
                response.redirect(VOICE_CONTINUE_URL)
            
            else:
                # Image not uploaded yet
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                if state["upload_poll_count"] < MAX_UPLOAD_POLL_COUNT:
                    gather = response.gather(
                        input="speech",
                        timeout=15,
                        speech_timeout="3",
                        action=VOICE_CONTINUE_URL,
                        method="POST"
                    )
                    gather.say(
                        "I don't see the upload yet. Please check your email for the link. "
                        "Let me know when you've uploaded the image, or say 'skip' to continue without it."
                    )
                    response.redirect(VOICE_CONTINUE_URL)
                else:
                    # Timeout - move to scheduling
                    state["step"] = "collect_zip"
                    update_state(call_sid, state)
                    
                    gather = response.gather(
                        input="speech",
                        timeout=5,
                        speech_timeout="3",
                        action=VOICE_CONTINUE_URL,
                        method="POST"
                    )
                    gather.say(
                        "We've been waiting a while. Let's continue with scheduling a technician. "
                        "You can still upload the photo later using the link in your email. "
                        "What is your ZIP code?"
                    )
                    response.redirect(VOICE_CONTINUE_URL)
        
        elif "skip" in text_lower or "schedule" in text_lower or "technician" in text_lower:
            # User wants to skip upload and go to scheduling
            state["step"] = "collect_zip"
            state["waiting_for_upload"] = False
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - User skipped upload, moving to scheduling")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "No problem. You can still upload the photo later using the email link. "
                "Let's schedule a technician. What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Check DB proactively for upload status
            upload_status = get_upload_status_by_call_sid(call_sid)
            
            if upload_status and upload_status.get("analysis_ready"):
                # Analysis ready - proceed to speak results
                state["step"] = "speak_analysis"
                update_state(call_sid, state)
                response.redirect(VOICE_CONTINUE_URL)
            
            else:
                # Still waiting - gentle prompt
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                if state["upload_poll_count"] < MAX_UPLOAD_POLL_COUNT:
                    gather = response.gather(
                        input="speech",
                        timeout=15,
                        speech_timeout="3",
                        action=VOICE_CONTINUE_URL,
                        method="POST"
                    )
                    gather.say(
                        "I'm still here. Let me know once you've uploaded the image, "
                        "or say 'skip' to schedule a technician instead."
                    )
                    response.redirect(VOICE_CONTINUE_URL)
                else:
                    # Max polls reached - fall back to scheduling
                    state["step"] = "collect_zip"
                    update_state(call_sid, state)
                    
                    gather = response.gather(
                        input="speech",
                        timeout=5,
                        speech_timeout="3",
                        action=VOICE_CONTINUE_URL,
                        method="POST"
                    )
                    gather.say(
                        "We've been waiting a while. Let's continue with scheduling. "
                        "What is your ZIP code?"
                    )
                    response.redirect(VOICE_CONTINUE_URL)
    
    # =========================================================================
    # ISSUE 2: Speak vision analysis results back to user
    # =========================================================================
    
    elif current_step == "speak_analysis":
        # Fetch analysis from DB
        upload_status = get_upload_status_by_call_sid(call_sid)
        
        if not upload_status or not upload_status.get("analysis_ready"):
            # No analysis available - shouldn't happen but handle gracefully
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "I'm sorry, the image analysis isn't available yet. "
                "Let's schedule a technician to take a look. What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        elif upload_status.get("is_appliance_image") == False:
            # ISSUE 2.4: Image is NOT an appliance - ask for re-upload
            appliance = state.get("appliance_type") or "appliance"
            state["step"] = "waiting_for_upload"
            state["upload_poll_count"] = 0  # Reset poll count for re-upload
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - Image was not an appliance, asking for re-upload")
            
            gather = response.gather(
                input="speech",
                timeout=15,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"The image doesn't appear to show the {appliance}. "
                "Please upload a clear photo of the appliance itself, "
                "especially showing any error codes or the problem area. "
                "Say 'done' when you've uploaded a new photo, or 'skip' to schedule a technician."
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Valid appliance image with analysis
            summary = upload_status.get("analysis_summary", "")
            tips = upload_status.get("troubleshooting_tips", "")
            
            state["image_analysis_spoken"] = True
            state["step"] = "after_analysis"
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - Speaking analysis results")
            
            # Build response with analysis
            analysis_speech = "Thanks, I've reviewed your image. "
            
            if summary:
                # Truncate if too long for speech
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                analysis_speech += summary + " "
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            
            if tips:
                # Truncate tips for speech
                if len(tips) > 200:
                    tips = tips[:197] + "..."
                analysis_speech += f"Here's what you can try: {tips} "
                analysis_speech += "Would you like to try this now and let me know if it helps? Or would you prefer to schedule a technician?"
            else:
                analysis_speech += "Based on what I see, I recommend scheduling a technician for a proper diagnosis. Would you like to schedule an appointment?"
            
            gather.say(analysis_speech)
            response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "after_analysis":
        # User responds to analysis - did they try the fix?
        text_lower = speech_result.lower()
        
        # Check for NEGATIVE responses FIRST - "not working", "didn't help", etc.
        negative_patterns = ["not work", "didn't work", "doesn't work", "don't work",
                           "didn't help", "doesn't help", "not help", "still broken",
                           "still not", "no luck", "same issue", "same problem"]
        is_negative = any(pattern in text_lower for pattern in negative_patterns)
        
        if is_negative or "schedule" in text_lower or "technician" in text_lower or "appointment" in text_lower or is_no_response(speech_result):
            # User says it didn't work OR wants technician
            print(f"[Tier 3] CallSid: {call_sid} - Troubleshooting didn't help, offering technician")
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            customer_name = state.get("customer_name", "")
            gather = response.gather(
                input="speech",
                timeout=8,
                speech_timeout="4",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"I'm sorry the troubleshooting didn't resolve the issue{', ' + customer_name if customer_name else ''}. "
                "Let me schedule a technician for you. What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        elif is_positive_response(speech_result) or "helped" in text_lower or "worked" in text_lower or "fixed" in text_lower or "better" in text_lower:
            # Issue resolved - only if clearly positive
            state["resolved"] = True
            state["step"] = "done"
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - Issue resolved after image analysis")
            
            response.say(
                "Great, I'm glad that helped! "
                "If the issue comes back, you can always call us again. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            response.hangup()
        
        else:
            # Unclear - ask again
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "Would you like to try the suggested fix, or would you prefer to schedule a technician?"
            )
            response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "collect_zip":
        # Extract ZIP code from speech (5 digits)
        digits = re.sub(r'\D', '', speech_result)
        if len(digits) >= 5:
            zip_code = digits[:5]
            state["zip_code"] = zip_code
            state["zip_attempts"] = 0  # Reset on success
            state["step"] = "collect_time_pref"
            update_state(call_sid, state)
            
            print(f"[State Update] CallSid: {call_sid}, ZIP: {zip_code}")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"Got it, ZIP code {' '.join(zip_code)}. "
                "Do you prefer a morning or afternoon appointment?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            # Track ZIP code attempts to prevent infinite loop
            state["zip_attempts"] = state.get("zip_attempts", 0) + 1
            update_state(call_sid, state)
            
            print(f"[Validation] ZIP attempt {state['zip_attempts']}/3, input: '{speech_result}'")
            
            if state["zip_attempts"] < 3:
                gather = response.gather(
                    input="speech",
                    timeout=8,
                    speech_timeout="4",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I'm sorry, I didn't catch a valid ZIP code. "
                    "Please say your 5-digit ZIP code clearly, like 6 0 6 0 1."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Max attempts - end call gracefully
                state["step"] = "done"
                update_state(call_sid, state)
                
                print(f"[Validation] ZIP capture failed after 3 attempts")
                
                response.say(
                    "I'm having trouble understanding the ZIP code. "
                    "Please visit our website or call back to schedule your appointment. "
                    "Thank you for calling Sears Home Services. Goodbye."
                )
                response.hangup()
    
    elif current_step == "collect_time_pref":
        text_lower = speech_result.lower()
        if "morning" in text_lower:
            time_pref = "morning"
        elif "afternoon" in text_lower or "evening" in text_lower:
            time_pref = "afternoon"
        else:
            time_pref = None  # No preference
        
        state["time_preference"] = time_pref
        
        print(f"[State Update] CallSid: {call_sid}, Time Preference: {time_pref}")
        
        # Find available slots
        slots = find_available_slots(
            zip_code=state.get("zip_code"),
            appliance_type=state.get("appliance_type"),
            time_preference=time_pref,
            limit=3
        )
        
        if not slots:
            state["step"] = "done"
            update_state(call_sid, state)
            
            response.say(
                "I'm sorry, we don't have any technicians available in your area "
                f"for {state.get('appliance_type')} service at this time. "
                "Please call back later or visit our website to schedule. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            response.hangup()
        else:
            state["offered_slots"] = slots
            state["step"] = "choose_slot"
            update_state(call_sid, state)
            
            # Build speech for available slots
            slot_speech = "Here are the available appointments: "
            for i, slot in enumerate(slots, 1):
                slot_speech += format_slot_for_speech(slot, i) + ". "
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                slot_speech +
                "Please say option 1, option 2, or option 3 to select your preferred time."
            )
            response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "choose_slot":
        text_lower = speech_result.lower()
        offered_slots = state.get("offered_slots", [])
        
        # Determine which option was selected
        chosen_index = None
        if "1" in text_lower or "one" in text_lower or "first" in text_lower:
            chosen_index = 0
        elif "2" in text_lower or "two" in text_lower or "second" in text_lower:
            chosen_index = 1
        elif "3" in text_lower or "three" in text_lower or "third" in text_lower:
            chosen_index = 2
        
        if chosen_index is not None and chosen_index < len(offered_slots):
            chosen_slot = offered_slots[chosen_index]
            
            # Get customer phone from form data (stored earlier or from Twilio)
            customer_phone = state.get("customer_phone", "")
            
            try:
                appt_info = book_appointment(
                    call_sid=call_sid,
                    customer_phone=customer_phone,
                    zip_code=state.get("zip_code"),
                    appliance_type=state.get("appliance_type"),
                    symptom_summary=state.get("symptom_summary", ""),
                    error_codes=state.get("error_codes", []),
                    is_urgent=state.get("is_urgent", False),
                    chosen_slot_id=chosen_slot["slot_id"]
                )
                
                state["step"] = "done"
                state["appointment_booked"] = True
                state["appointment_id"] = appt_info["id"]
                update_state(call_sid, state)
                
                print(f"[Appointment Booked] CallSid: {call_sid}, Appointment ID: {appt_info['id']}")
                
                # Format confirmation
                start = appt_info["start_time"]
                day_name = start.strftime("%A")
                date_str = start.strftime("%B %d")
                hour = start.hour
                if hour < 12:
                    time_str = f"{hour} AM" if hour > 0 else "12 AM"
                else:
                    hour_12 = hour - 12 if hour > 12 else hour
                    time_str = f"{hour_12} PM" if hour_12 > 0 else "12 PM"
                
                response.say(
                    f"Your appointment is confirmed for {day_name}, {date_str} at {time_str} "
                    f"with technician {appt_info['technician_name']}. "
                    "You will receive a confirmation text shortly. "
                    "Thank you for calling Sears Home Services. Goodbye."
                )
                response.hangup()
                
            except Exception as e:
                print(f"[Booking Error] CallSid: {call_sid}, Error: {e}")
                state["step"] = "done"
                update_state(call_sid, state)
                
                response.say(
                    "I'm sorry, there was an error booking your appointment. "
                    "Please call back or visit our website to schedule. "
                    "Thank you for calling Sears Home Services. Goodbye."
                )
                response.hangup()
        else:
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "I'm sorry, I didn't understand your selection. "
                "Please say option 1, option 2, or option 3."
            )
            response.redirect(VOICE_CONTINUE_URL)
    
    else:
        response.say("I'm sorry, something went wrong. Please call back later. Goodbye.")
        response.hangup()
    
    return Response(content=str(response), media_type="application/xml")
