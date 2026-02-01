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
from .llm import llm_classify_appliance, llm_extract_symptoms
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


def extract_email_from_speech(speech_text: str) -> str:
    """
    ISSUE 1: Improved email extraction from speech-to-text.
    
    Handles two scenarios:
    1. Letter-by-letter spelling: "K A S I dot M A J J I at gmail dot com"
    2. Full email spoken: "kasi.majji at gmail dot com"
    
    Normalizes by:
    - Converting spoken words to symbols (" at " → @, " dot " → .)
    - Removing spaces between letters
    - Lowercasing everything
    """
    text = speech_text.lower().strip()
    
    # Stage 1: Replace spoken words with symbols (before removing spaces)
    # Order matters - do longer phrases first
    replacements = [
        # Common TLD spoken forms
        ("dot com", ".com"),
        ("dot net", ".net"),
        ("dot org", ".org"),
        ("dot edu", ".edu"),
        ("dot co dot uk", ".co.uk"),
        ("dot co", ".co"),
        # At symbol variations
        ("at symbol", "@"),
        ("at sign", "@"),
        (" at ", "@"),
        # Dot variations
        (" dot ", "."),
        (" period ", "."),
        (" point ", "."),
        # Other symbols
        (" underscore ", "_"),
        (" dash ", "-"),
        (" hyphen ", "-"),
        # Common domains (preserve as-is)
        ("gmail", "gmail"),
        ("yahoo", "yahoo"),
        ("hotmail", "hotmail"),
        ("outlook", "outlook"),
        ("icloud", "icloud"),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Stage 2: Handle letter-by-letter spelling
    # If there are single letters separated by spaces, join them
    # e.g., "k a s i" → "kasi"
    words = text.split()
    result_parts = []
    letter_buffer = []
    
    for word in words:
        # Check if it's a single letter or symbol
        if len(word) == 1 and word.isalpha():
            letter_buffer.append(word)
        else:
            # Flush letter buffer
            if letter_buffer:
                result_parts.append("".join(letter_buffer))
                letter_buffer = []
            result_parts.append(word)
    
    # Flush remaining letters
    if letter_buffer:
        result_parts.append("".join(letter_buffer))
    
    # Join without spaces
    text = "".join(result_parts)
    
    # Stage 3: Clean up any remaining issues
    text = text.replace(" ", "")  # Remove any remaining spaces
    
    # Try to find an email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    
    if match:
        return match.group(0)
    
    return text


def spell_email_for_speech(email: str) -> str:
    """
    ISSUE 1: Convert email to speakable format for confirmation.
    
    Example: "kasi.majji@gmail.com" → "k a s i dot m a j j i at g m a i l dot com"
    """
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
        elif char.isalnum():
            result.append(f" {char} ")
        else:
            result.append(f" {char} ")
    
    # Clean up extra spaces
    return " ".join(result.split())


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
    state["step"] = "ask_appliance"
    state["customer_phone"] = from_number
    update_state(call_sid, state)
    
    response = VoiceResponse()
    
    response.say(
        "Hi, this is the Sears Home Services assistant demo. "
        "Thanks for calling. I'll help you troubleshoot your appliance issue."
    )
    
    gather = response.gather(
        input="speech",
        timeout=5,
        speech_timeout="3",
        action=VOICE_CONTINUE_URL,
        method="POST"
    )
    gather.say(
        "To get started, what appliance are you calling about? "
        "For example, a washer, dryer, or refrigerator."
    )
    
    response.redirect(VOICE_CONTINUE_URL)
    
    twiml_string = str(response)
    return Response(content=twiml_string, media_type="application/xml")


@router.post("/voice/continue")
async def voice_continue(request: Request):
    """Handles the response after the user speaks - implements state machine."""
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid", "")
    speech_result = form_data.get("SpeechResult", "")
    
    print(f"[Speech Received] CallSid: {call_sid}, SpeechResult: {speech_result}")
    
    state = get_state(call_sid)
    response = VoiceResponse()
    
    speech_result = speech_result or ""
    
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
    
    current_step = state.get("step", "ask_appliance")
    
    if current_step == "ask_appliance":
        # Try Gemini LLM first, fall back to keyword-based inference
        appliance = llm_classify_appliance(speech_result)
        if not appliance:
            appliance = infer_appliance_type(speech_result)
            if appliance:
                print(f"[Fallback] Using keyword-based appliance detection: {appliance}")
        
        if appliance:
            state["appliance_type"] = appliance
            state["step"] = "ask_symptoms"
            state["no_match_attempts"] = 0  # Reset on success
            update_state(call_sid, state)
            
            print(f"[State Update] CallSid: {call_sid}, Appliance: {appliance}")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"Thanks. I heard {appliance}. "
                f"Now, can you briefly describe what's going wrong with your {appliance}?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            state["no_match_attempts"] = state.get("no_match_attempts", 0) + 1
            update_state(call_sid, state)
            
            if state["no_match_attempts"] <= 2:
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I'm sorry, I didn't catch what appliance you're calling about. "
                    "Please say something like washer, dryer, refrigerator, dishwasher, oven, or HVAC system."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                response.say(
                    "I'm still having trouble understanding the appliance type, "
                    "so I'll end the call here. "
                    "You can also schedule service online at any time. Goodbye."
                )
                response.hangup()
    
    elif current_step == "ask_symptoms":
        # Store raw symptoms
        state["symptoms"] = speech_result
        
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
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            # Use the LLM-generated symptom summary in the response
            summary = state["symptom_summary"]
            gather.say(
                f"Thanks. It sounds like you're experiencing: {summary}. "
                f"Let's try a quick check together. {prompt} "
                "After you've checked that, just say 'yes' if it helped or 'no' if the problem is still there."
            )
            response.redirect(VOICE_CONTINUE_URL)
        else:
            state["step"] = "done"
            update_state(call_sid, state)
            response.say(
                "I'm sorry, I don't have troubleshooting steps for that appliance yet. "
                "Please call back to speak with a technician. Goodbye."
            )
            response.hangup()
    
    elif current_step == "troubleshoot":
        if is_positive_response(speech_result):
            state["resolved"] = True
            state["step"] = "done"
            update_state(call_sid, state)
            
            print(f"[Call Resolved] CallSid: {call_sid}")
            
            response.say(
                "Great, I'm glad that seemed to help! "
                "If the issue comes back, you can always call us again. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            response.hangup()
        else:
            prompt = get_next_troubleshooting_prompt(state)
            update_state(call_sid, state)
            
            if prompt:
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    f"Okay, let's try another check. {prompt} "
                    "After you've checked that, just say 'yes' if it helped or 'no' if the problem is still there."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Offer Tier 3 image upload option before scheduling
                state["step"] = "offer_image_upload"
                update_state(call_sid, state)
                
                print(f"[Escalation Needed] CallSid: {call_sid} - Offering Tier 3 image upload")
                
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I wasn't able to resolve the issue with basic troubleshooting. "
                    "I have two options for you. "
                    "First, I can send you a link to upload a photo of your appliance "
                    "for additional AI-powered diagnosis. "
                    "Or second, I can help you schedule a technician visit. "
                    "Would you like to upload a photo, or schedule a technician?"
                )
                response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "offer_image_upload":
        text_lower = speech_result.lower()
        
        if "photo" in text_lower or "picture" in text_lower or "image" in text_lower or "upload" in text_lower:
            # User wants Tier 3 image upload
            state["step"] = "collect_email"
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - User chose image upload")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "Great! I'll send you an upload link by email. "
                "Please say your email address slowly and clearly."
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        elif "technician" in text_lower or "schedule" in text_lower or "appointment" in text_lower or "visit" in text_lower:
            # User wants Tier 2 scheduling
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            print(f"[Tier 2] CallSid: {call_sid} - User chose technician scheduling")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "Let me help you schedule a technician visit. "
                "What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Unclear response, ask again
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
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
            state["email_confirm_attempts"] = 0
            update_state(call_sid, state)
            
            # Spell back the email for confirmation
            spelled_email = spell_email_for_speech(email)
            
            print(f"[Tier 3] CallSid: {call_sid} - Email captured: {email}, awaiting confirmation")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"I heard {spelled_email}. "
                "Is that correct? Please say yes or no."
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            state["email_attempts"] = state.get("email_attempts", 0) + 1
            update_state(call_sid, state)
            
            if state["email_attempts"] <= 3:
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I'm sorry, I didn't catch a valid email address. "
                    "Please say your email slowly. You can spell it out letter by letter, "
                    "like k a s i dot m a j j i at gmail dot com."
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
                    method="POST"
                )
                gather.say(
                    "I'm having trouble capturing the email. "
                    "I'll skip image upload and continue with scheduling. "
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
            
            if state["email_confirm_attempts"] <= 2:
                state["step"] = "collect_email"
                update_state(call_sid, state)
                
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "No problem, let's try again. "
                    "Please say your email address slowly and clearly."
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
                    method="POST"
                )
                gather.say(
                    "I'm having trouble with the email. "
                    "I'll skip image upload and continue with scheduling. "
                    "What is your ZIP code?"
                )
                response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # Unclear response, ask again
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
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
        
        if is_positive_response(speech_result) or "try" in text_lower or "helped" in text_lower or "worked" in text_lower or "fixed" in text_lower:
            # Issue resolved
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
        
        elif "schedule" in text_lower or "technician" in text_lower or "appointment" in text_lower or is_no_response(speech_result):
            # User wants technician
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
                "Let me schedule a technician for you. What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
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
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                "I'm sorry, I didn't catch a valid ZIP code. "
                "Please say your 5-digit ZIP code."
            )
            response.redirect(VOICE_CONTINUE_URL)
    
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
