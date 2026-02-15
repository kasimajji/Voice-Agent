import re
import time
from fastapi import APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Say

from .config import (
    APP_BASE_URL, TTS_VOICE, STT_SPEECH_MODEL,
    USE_STREAMING_STT, STT_CONFIDENCE_THRESHOLD,
    AUTONOMOUS_AGENT_MODE,
    get_base_url_from_request,
)
from .stt_stream import get_transcript, clear_transcript, wait_for_transcript
from .conversation import (
    get_state,
    update_state,
    infer_appliance_type,
)
from .llm import (
    llm_classify_appliance,
    llm_extract_symptoms,
    llm_is_appliance_related,
    llm_extract_email,
    llm_extract_name,
    llm_analyze_customer_intent,
    llm_plan_next_step,
    llm_interpret_troubleshooting_response,
    llm_classify_yes_no,
    llm_classify_user_intent,
    llm_extract_zip_code,
    llm_extract_time_preference,
    llm_choose_slot,
    llm_interpret_upload_intent,
    llm_interpret_after_analysis,
    llm_generate_troubleshooting_steps,
)
from .scheduling import find_available_slots, book_appointment, format_slot_for_speech
from .image_service import (
    create_image_upload_token,
    build_upload_url,
    send_upload_email,
    validate_email,
    get_upload_status_by_call_sid,
    reset_upload_for_reupload,
)
from .logging_config import (
    get_logger,
    log_conversation,
    log_state_change,
    log_call_start,
    log_call_end,
    log_error,
)

logger = get_logger("twilio")

router = APIRouter()

# Maximum polling attempts for image upload
MAX_UPLOAD_POLL_COUNT = 10  # ~2.5 minutes with 15s pauses


def _get_continue_url(request: Request) -> str:
    """
    Compute the continue URL dynamically from the request Host header.
    Solves ngrok URL drift: twilio-config updates Twilio with the live ngrok URL,
    but APP_BASE_URL computed at import time may be stale.
    """
    base = get_base_url_from_request(request)
    return f"{base}/twilio/voice/continue"


def _get_stream_url(request: Request) -> str:
    """Compute the WebSocket URL for Twilio Media Streams."""
    base = get_base_url_from_request(request)
    # Convert https:// to wss:// for WebSocket
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/twilio/media-stream"


def _add_media_stream(response: VoiceResponse, stream_url: str):
    """
    Add a <Start><Stream> element to the TwiML response.
    This starts real-time audio streaming to our Google STT endpoint
    alongside the normal Gather-based flow.
    """
    if USE_STREAMING_STT:
        start = response.start()
        start.stream(url=stream_url)


def _build_gather(response: VoiceResponse, action_url: str, timeout: int = 5,
                  speech_timeout: str = "3", hints: str = None,
                  language: str = None) -> object:
    """
    Build a Gather element with consistent STT settings across all steps.
    Uses speech_model='phone_call' for optimized telephony recognition.
    bargeIn is always False per requirement.
    """
    kwargs = {
        "input": "speech",
        "timeout": timeout,
        "speech_timeout": speech_timeout,
        "action": action_url,
        "method": "POST",
        "bargeIn": False,
        "speechModel": STT_SPEECH_MODEL,
    }
    if hints:
        kwargs["hints"] = hints
    if language:
        kwargs["language"] = language
    return response.gather(**kwargs)


def create_ssml_say(text: str, voice: str = "default", rate: str = "normal") -> Say:
    """
    Create a Say object with consistent Neural voice.
    Uses Polly.Joanna-Neural for natural-sounding speech.
    """
    return Say(text, voice=TTS_VOICE)


def say_with_logging(text: str, call_sid: str = "", step: str = None,
                     voice: str = "default", rate: str = "normal"):
    """Create Say object with logging."""
    log_conversation(call_sid, "AGENT", text, step)
    return create_ssml_say(text)


def extract_email_from_speech(speech_text: str, call_sid: str = "") -> str:
    """
    Extract email from Twilio speech-to-text using AI.
    
    Delegates to llm_extract_email for intelligent extraction that handles:
    - Spelled out letters: "k a s i at gmail dot com"
    - Twilio artifacts: "K. A s. I dot m. A j. J. I at gmail.com."
    - Common phrasings: "at the rate", "dot com"
    
    Returns:
        Extracted or constructed email string. Never returns None.
    """
    log_conversation(call_sid, "EMAIL_EXTRACT", f"Raw input: {speech_text}", "collect_email")
    email = llm_extract_email(speech_text)
    log_conversation(call_sid, "EMAIL_EXTRACT", f"LLM result: {email}", "collect_email")
    return email


def speak_email_naturally(email: str) -> str:
    """
    Convert email to a natural spoken form first, then spell it out.
    
    Example: "kasi.majji@gmail.com" →
      "kasi dot majji at gmail dot com. Let me spell that out: k, a, s, i, dot, m, a, j, j, i, at, g, m, a, i, l, dot, c, o, m."
    
    This addresses the recruiter feedback: after extracting the email, speak it
    naturally first (how humans say it), then slowly spell it for verification.
    """
    if not email:
        return ""
    
    # Part 1: Natural pronunciation
    # Split into username and domain
    parts = email.lower().split("@")
    if len(parts) != 2:
        return _spell_email_slow(email)
    
    username = parts[0]
    domain = parts[1]
    
    # Pronounce username naturally (replace dots/underscores with words)
    natural_user = username.replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
    natural_domain = domain.replace(".", " dot ")
    natural_form = f"{natural_user} at {natural_domain}"
    
    # Part 2: Slow spelled-out version
    spelled = _spell_email_slow(email)
    
    return (
        f"I heard your email as {natural_form}. "
        f"Let me spell that out to make sure: {spelled}. "
        "Is that correct? Please say yes or no."
    )


def _spell_email_slow(email: str) -> str:
    """Spell email character by character with pauses for clarity."""
    if not email:
        return ""
    
    result = []
    for char in email.lower():
        if char == '@':
            result.append("at")
        elif char == '.':
            result.append("dot")
        elif char == '_':
            result.append("underscore")
        elif char == '-':
            result.append("dash")
        elif char == '+':
            result.append("plus")
        elif char.isalnum():
            result.append(char)
    
    spelled = ", ".join(result)
    logger.debug(f"Email spelled: {email} → {spelled}")
    return spelled


# NOTE: is_yes_response / is_no_response removed — all confirmation
# logic is now handled by llm_classify_yes_no for 100% AI autonomy.


@router.post("/voice")
async def voice_entry(request: Request):
    """Entry point when a call starts - Twilio hits this webhook."""
    form_data = await request.form()
    continue_url = _get_continue_url(request)
    
    call_sid = form_data.get("CallSid", "")
    from_number = form_data.get("From", "")
    to_number = form_data.get("To", "")
    
    log_call_start(call_sid, from_number, to_number)
    
    state = get_state(call_sid)
    state["step"] = "greet_ask_name"
    state["customer_phone"] = from_number
    update_state(call_sid, state)
    
    response = VoiceResponse()
    
    # Start real-time audio streaming to Google STT (alongside Gather fallback)
    stream_url = _get_stream_url(request)
    _add_media_stream(response, stream_url)
    
    # Natural greeting - warm and friendly
    greeting_text = (
        "Hi there! Thanks for calling Sears Home Services. "
        "My name is Sam, and I'll be helping you today. "
        "May I have your name, please?"
    )
    log_conversation(call_sid, "AGENT", greeting_text, "greet_ask_name")
    
    gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
    gather.append(create_ssml_say(greeting_text))
    
    response.redirect(continue_url)
    
    return Response(content=str(response), media_type="application/xml")


@router.post("/voice/continue")
async def voice_continue(request: Request):
    """Handles the response after the user speaks - implements state machine."""
    call_sid = ""
    try:
        form_data = await request.form()
        continue_url = _get_continue_url(request)
        
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")
        twilio_confidence = form_data.get("Confidence", "")
        
        state = get_state(call_sid)
        current_step = state.get("step", "unknown")
        
        # ── Confidence gating ──
        # Reject low-confidence Twilio transcripts as noise (fixes phantom captures)
        # BUT skip gating for steps that expect short/numeric input — those naturally
        # have lower Twilio confidence scores and should not be rejected.
        _skip_gating_steps = {
            "collect_zip", "confirm_zip", "confirm_email", "confirm_resolution",
            "choose_slot", "collect_time_pref", "offer_troubleshoot_or_schedule",
            "offer_image_upload", "after_analysis",
        }
        if speech_result and twilio_confidence and current_step not in _skip_gating_steps:
            try:
                conf = float(twilio_confidence)
                if conf < STT_CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"Rejected low-confidence speech: '{speech_result}' (conf={conf:.2f})",
                        extra={"call_sid": call_sid},
                    )
                    speech_result = ""
            except ValueError:
                pass
        turn_start = state.get("_turn_start_ts", 0.0)
        
        # ── Smart STT selection: pick the BEST transcript ──
        # Compare Google STT streaming vs Twilio STT and pick the winner based on
        # confidence. Neither is universally better: Twilio excels at short/numeric
        # utterances (ZIP codes, yes/no), Google STT excels at longer sentences.
        if USE_STREAMING_STT and call_sid:
            stream_transcript = get_transcript(call_sid, not_before=turn_start)
            if stream_transcript and stream_transcript.get("text"):
                stream_text = stream_transcript["text"]
                stream_conf = stream_transcript.get("confidence", 0.0)
                is_final = stream_transcript.get("is_final", False)
                
                # Parse Twilio confidence for comparison
                twilio_conf = 0.0
                try:
                    twilio_conf = float(twilio_confidence) if twilio_confidence else 0.0
                except (ValueError, TypeError):
                    pass
                
                if not speech_result.strip():
                    # Twilio returned nothing — use Google STT (rescue)
                    logger.info(
                        f"[STT rescue] Google='{stream_text[:60]}' (conf={stream_conf:.2f}, final={is_final})",
                        extra={"call_sid": call_sid},
                    )
                    speech_result = stream_text
                elif is_final and stream_conf > twilio_conf:
                    # Google STT has higher confidence — prefer it
                    logger.info(
                        f"[STT pick Google] '{stream_text[:60]}' (conf={stream_conf:.2f}) "
                        f"over Twilio: '{speech_result[:60]}' (conf={twilio_conf:.2f})",
                        extra={"call_sid": call_sid},
                    )
                    speech_result = stream_text
                else:
                    # Twilio has equal or higher confidence — keep it
                    logger.info(
                        f"[STT pick Twilio] '{speech_result[:60]}' (conf={twilio_conf:.2f}) "
                        f"over Google: '{stream_text[:60]}' (conf={stream_conf:.2f})",
                        extra={"call_sid": call_sid},
                    )
            clear_transcript(call_sid)
        
        # Log customer speech
        log_conversation(call_sid, "CUSTOMER", speech_result or "(silence)", current_step)

        # ── Autonomous planner routing (feature-flagged) ──
        # In autonomous mode, we compute the next step from goal-grounded
        # intent/state planning instead of relying solely on static transitions.
        if AUTONOMOUS_AGENT_MODE:
            planned_step = llm_plan_next_step(speech_result, state)
            if planned_step and planned_step != state.get("step"):
                logger.info(
                    f"[Autonomous planner] step {state.get('step')} -> {planned_step}",
                    extra={"call_sid": call_sid},
                )
                state["step"] = planned_step
                update_state(call_sid, state)
        
        response = VoiceResponse()
        
        speech_result = speech_result or ""
        
        # Main state machine logic wrapped in inner try for graceful degradation
        return await _handle_voice_continue(call_sid, speech_result, state, response, continue_url)
        
    except Exception as e:
        # Critical error handler - ensures call never crashes silently
        log_error(call_sid, e, step="voice_continue", context="Critical error in voice handler")
        response = VoiceResponse()
        response.append(create_ssml_say(
            "I'm sorry, we're experiencing technical difficulties. "
            "Please call back in a few minutes. Goodbye."
        ))
        response.hangup()
        return Response(content=str(response), media_type="application/xml")


async def _handle_voice_continue(call_sid: str, speech_result: str, state: dict,
                                  response: VoiceResponse, continue_url: str):
    """Inner handler for voice_continue - separated for cleaner error handling."""
    
    current_step = state.get("step", "greet_ask_name")
    customer_name = state.get("customer_name", "")
    name_phrase = f", {customer_name}" if customer_name else ""

    # Explicit terminal handler for autonomous planner routing.
    # Without this, planner-selected "done" can fall through to the generic
    # error branch and produce a confusing system message.
    if current_step == "done":
        logger.info("Handling terminal done step", extra={"call_sid": call_sid, "step": "done"})
        # Use LLM to detect callback intent for personalized goodbye
        if speech_result and speech_result.strip():
            cb_result = llm_classify_user_intent(
                speech_result,
                choices=["callback", "other"],
                context="The call is ending. Did the customer say they want to call back later?"
            )
            is_callback = cb_result.get("choice") == "callback" and cb_result.get("confidence", 0) >= 0.6
        else:
            is_callback = False
        
        if is_callback:
            goodbye_text = (
                f"No problem{name_phrase}. You can call us back anytime when you're ready. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
        elif state.get("appointment_booked"):
            goodbye_text = "Thank you for calling Sears Home Services. Goodbye."
        elif state.get("resolved"):
            goodbye_text = "Great, glad we could help. Thank you for calling Sears Home Services. Goodbye."
        else:
            goodbye_text = "Thank you for calling Sears Home Services. Goodbye."

        say_obj = say_with_logging(goodbye_text, call_sid, "done")
        response.append(say_obj)
        response.hangup()
        return Response(content=str(response), media_type="application/xml")
    
    # ==================== NO-INPUT HANDLING ====================
    # speak_analysis arrives via redirect (no speech expected) — skip no-input entirely,
    # let the step handler below speak the analysis results
    if not speech_result.strip() and current_step != "speak_analysis":
        state["no_input_attempts"] = state.get("no_input_attempts", 0) + 1
        update_state(call_sid, state)
        
        # Special handling for waiting_for_upload - check if image was uploaded automatically
        if current_step == "waiting_for_upload":
            upload_status = get_upload_status_by_call_sid(call_sid)
            
            if upload_status and upload_status.get("analysis_ready"):
                state["step"] = "speak_analysis"
                state["waiting_for_upload"] = False
                state["no_input_attempts"] = 0
                update_state(call_sid, state)
                logger.info("Auto-detected image upload, speaking results", extra={"call_sid": call_sid, "step": "waiting_for_upload"})
                response.redirect(continue_url)
                return Response(content=str(response), media_type="application/xml")
            
            elif upload_status and upload_status.get("image_uploaded"):
                state["no_input_attempts"] = 0
                update_state(call_sid, state)
                gather = _build_gather(response, continue_url, timeout=10, speech_timeout="3")
                agent_text = "I see your image was received. Just a moment while I analyze it."
                log_conversation(call_sid, "AGENT", agent_text, "waiting_for_upload")
                gather.append(create_ssml_say(agent_text))
                response.pause(length=3)
                response.redirect(continue_url)
                return Response(content=str(response), media_type="application/xml")
            
            else:
                upload_wait_attempts = state.get("upload_wait_attempts", 0) + 1
                state["upload_wait_attempts"] = upload_wait_attempts
                update_state(call_sid, state)
                
                if upload_wait_attempts <= 2:
                    gather = _build_gather(response, continue_url, timeout=20, speech_timeout="3")
                    agent_text = (
                        "I'm still here waiting for your upload. "
                        "Do you need more time? Just say yes if you need more time, "
                        "or skip if you'd like to schedule a technician instead."
                    )
                    log_conversation(call_sid, "AGENT", agent_text, "waiting_for_upload")
                    gather.append(create_ssml_say(agent_text))
                    response.redirect(continue_url)
                else:
                    state["step"] = "collect_zip"
                    state["upload_wait_attempts"] = 0
                    state["no_input_attempts"] = 0
                    update_state(call_sid, state)
                    
                    gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                    agent_text = (
                        "No worries, let's schedule a technician to help you in person. "
                        "You can still upload the photo later using the link in your email. "
                        "What is your ZIP code?"
                    )
                    log_conversation(call_sid, "AGENT", agent_text, "collect_zip")
                    gather.append(create_ssml_say(agent_text))
                    response.redirect(continue_url)
                
                return Response(content=str(response), media_type="application/xml")
        
        # Normal no-input handling for other steps
        if state["no_input_attempts"] <= 2:
            no_input_text = "I'm sorry, I didn't hear anything. Please say that again."
            log_conversation(call_sid, "AGENT", no_input_text, "no_input")
            
            gather = _build_gather(response, continue_url, timeout=4, speech_timeout="2")
            gather.append(create_ssml_say(no_input_text))
            response.redirect(continue_url)
        else:
            state["step"] = "collect_zip"
            state["no_input_attempts"] = 0
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            agent_text = (
                "I'm having trouble hearing you. Let me help you schedule a technician. "
                "What is your ZIP code?"
            )
            log_conversation(call_sid, "AGENT", agent_text, "collect_zip")
            gather.append(create_ssml_say(agent_text))
            response.redirect(continue_url)
        
        return Response(content=str(response), media_type="application/xml")
    
    # Reset no-input counter on any valid speech
    state["no_input_attempts"] = 0
    
    # ==================== AUTONOMOUS CONVERSATION FLOW ====================
    
    if current_step == "greet_ask_name":
        # Use LLM to extract name accurately from speech
        customer_name = llm_extract_name(speech_result)
        
        state["customer_name"] = customer_name
        # Skip "how are you" — go directly to open-ended "how can I help"
        state["step"] = "understand_need"
        update_state(call_sid, state)
        
        logger.info(f"Customer name captured: {customer_name}", extra={"call_sid": call_sid, "step": "greet_ask_name"})
        
        # Combined greeting + open-ended question — customer can say anything
        if customer_name:
            greeting_text = (
                f"Nice to meet you, {customer_name}! "
                "How can I help you today?"
            )
        else:
            greeting_text = (
                "Thanks for calling! How can I help you today?"
            )
        log_conversation(call_sid, "AGENT", greeting_text, "greet_ask_name")
        
        gather = _build_gather(response, continue_url, timeout=10, speech_timeout="auto")
        gather.append(create_ssml_say(greeting_text))
        response.redirect(continue_url)
    
    # ==================== AUTONOMOUS INTENT DETECTION ====================
    # This is the core of the new flow. The customer can say anything:
    # - "My fridge is not cooling" → detect appliance + symptoms, offer troubleshooting
    # - "I want to schedule a technician" → skip to scheduling
    # - "My washer is making a loud noise and leaking water" → full description, skip symptom asking
    # - "I have a problem" → ask for more details
    
    elif current_step == "understand_need":
        # Use LLM to analyze the customer's intent from their open-ended response
        intent_result = llm_analyze_customer_intent(speech_result)
        
        logger.info(f"Intent analysis: {intent_result}", extra={"call_sid": call_sid, "step": "understand_need"})
        
        appliance = intent_result.get("appliance_type")
        symptoms = intent_result.get("symptoms")
        wants_scheduling = intent_result.get("wants_scheduling", False)
        has_full_description = intent_result.get("has_full_description", False)
        intent = intent_result.get("intent", "unclear")
        
        if appliance:
            state["appliance_type"] = appliance
        if symptoms:
            state["symptoms"] = speech_result
            state["symptom_summary"] = symptoms
        
        # CASE 1: Customer wants to schedule directly — skip everything
        if wants_scheduling and appliance:
            state["step"] = "collect_zip"
            if not state.get("symptom_summary"):
                state["symptom_summary"] = symptoms or speech_result
            update_state(call_sid, state)
            
            logger.info(f"Direct scheduling requested for {appliance}", extra={"call_sid": call_sid})
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Absolutely{name_phrase}! I'll help you schedule a technician for your {appliance}. "
                "What is your ZIP code?"
            ))
            response.redirect(continue_url)
        
        # CASE 2: Customer gave full problem description — offer troubleshooting or scheduling
        elif appliance and has_full_description:
            # Extract structured symptoms
            extracted = llm_extract_symptoms(speech_result)
            summary = extracted.get("symptom_summary") or symptoms or speech_result
            # Filter out 3rd-person meta-text from LLM
            summary_lower = summary.lower()
            meta_patterns = [
                "the caller", "the customer", "the user",
                "customer reported", "caller described", "user said",
                "customer's ", "caller's ", "user's ",
                "no error codes", "no specific", "no further",
                "reported that", "describes a", "mentioned that",
            ]
            if any(p in summary_lower for p in meta_patterns):
                summary = f"Your {appliance} is not working properly"
            state["symptom_summary"] = summary
            state["error_codes"] = extracted.get("error_codes") or []
            state["is_urgent"] = bool(extracted.get("is_urgent"))
            state["step"] = "offer_troubleshoot_or_schedule"
            update_state(call_sid, state)
            
            logger.info(f"Full description for {appliance}: {state['symptom_summary'][:80]}", extra={"call_sid": call_sid})
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Got it{name_phrase}. {summary}. "
                "Would you like me to walk you through a few quick troubleshooting steps, "
                "or would you prefer to schedule a technician right away?"
            ))
            response.redirect(continue_url)
        
        # CASE 3: Customer mentioned appliance but not enough detail
        elif appliance and not has_full_description:
            state["step"] = "ask_symptoms"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=10, speech_timeout="5")
            gather.append(create_ssml_say(
                f"Got it{name_phrase}, so you're having trouble with your {appliance}. "
                "Can you tell me a bit more about what's happening? "
                "For example, any error codes, strange noises, or specific issues you've noticed?"
            ))
            response.redirect(continue_url)
        
        # CASE 4: Wants scheduling but no appliance mentioned
        elif wants_scheduling:
            state["step"] = "ask_appliance_for_scheduling"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
            gather.append(create_ssml_say(
                f"Sure{name_phrase}, I can help you schedule a technician. "
                "Which appliance do you need help with? "
                "For example, a washer, dryer, refrigerator, dishwasher, oven, or HVAC system?"
            ))
            response.redirect(continue_url)
        
        # CASE 5: Unclear — ask for more details
        else:
            state["understand_attempts"] = state.get("understand_attempts", 0) + 1
            update_state(call_sid, state)
            
            if state["understand_attempts"] <= 2:
                gather = _build_gather(response, continue_url, timeout=10, speech_timeout="auto")
                gather.append(create_ssml_say(
                    f"I'd love to help{name_phrase}! Could you tell me which appliance is giving you trouble "
                    "and what's happening with it? For example, you could say "
                    "'my refrigerator is not cooling' or 'I need to schedule a washer repair'."
                ))
                response.redirect(continue_url)
            else:
                # After retries, ask directly
                state["step"] = "ask_appliance_for_scheduling"
                state["understand_attempts"] = 0
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
                gather.append(create_ssml_say(
                    "No problem! Which appliance do you need help with? "
                    "A washer, dryer, refrigerator, dishwasher, oven, or HVAC system?"
                ))
                response.redirect(continue_url)
    
    elif current_step == "ask_appliance_for_scheduling":
        # Customer wants scheduling but we need to know the appliance
        appliance = llm_classify_appliance(speech_result)
        if not appliance:
            appliance = infer_appliance_type(speech_result)
        
        if appliance:
            state["appliance_type"] = appliance
            state["step"] = "collect_zip"
            # Also try to extract any symptoms mentioned
            if not state.get("symptom_summary"):
                state["symptom_summary"] = speech_result
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Got it, {appliance} service. What is your ZIP code?"
            ))
            response.redirect(continue_url)
        else:
            state["appliance_attempts"] = state.get("appliance_attempts", 0) + 1
            update_state(call_sid, state)
            
            if state["appliance_attempts"] < 2:
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
                gather.append(create_ssml_say(
                    "I'm sorry, I didn't catch that. Which appliance needs service? "
                    "A washer, dryer, fridge, dishwasher, oven, or HVAC system?"
                ))
                response.redirect(continue_url)
            else:
                state["appliance_type"] = "appliance"
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                gather.append(create_ssml_say(
                    "No worries, our technician can help with any appliance. "
                    "What is your ZIP code?"
                ))
                response.redirect(continue_url)
    
    # ==================== TROUBLESHOOT OR SCHEDULE CHOICE ====================
    
    elif current_step == "offer_troubleshoot_or_schedule":
        # 100% LLM-powered intent classification
        llm_result = llm_classify_user_intent(
            speech_result,
            choices=["troubleshoot", "schedule", "callback"],
            context="Agent asked: Would you like me to walk you through troubleshooting steps, or schedule a technician?"
        )
        choice = llm_result.get("choice", "unclear")
        conf = llm_result.get("confidence", 0.0)
        
        logger.debug(f"Troubleshoot/schedule LLM: {llm_result}", extra={"call_sid": call_sid, "step": "offer_troubleshoot_or_schedule"})
        
        # Default to troubleshooting if unclear (most helpful action)
        if choice == "unclear" or conf < 0.5:
            choice = "troubleshoot"
        
        # Execute the choice
        if choice == "callback":
            state["step"] = "done"
            update_state(call_sid, state)

            goodbye_text = (
                f"No worries{name_phrase}. Feel free to reach out anytime if you need help. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            say_obj = say_with_logging(goodbye_text, call_sid, "offer_troubleshoot_or_schedule")
            response.append(say_obj)
            response.hangup()
            return Response(content=str(response), media_type="application/xml")

        elif choice == "schedule":
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Absolutely{name_phrase}! Let's get a technician out to you. "
                "What is your ZIP code?"
            ))
            response.redirect(continue_url)
        
        else:  # troubleshoot
            state["step"] = "troubleshoot_all"
            state["troubleshooting_step"] = 0
            update_state(call_sid, state)
            
            appliance = state.get("appliance_type", "appliance")
            symptom = state.get("symptom_summary", "")
            # Use LLM to generate context-aware troubleshooting steps
            steps_summary = llm_generate_troubleshooting_steps(appliance, symptom)
            state["troubleshooting_steps_text"] = steps_summary
            update_state(call_sid, state)
            
            if steps_summary:
                gather = _build_gather(response, continue_url, timeout=15, speech_timeout="4")
                agent_text = (
                    f"Alright{name_phrase}, here are a few quick things you can check: "
                    f"{steps_summary} "
                    "Please try these and let me know if any of them helped."
                )
                log_conversation(call_sid, "AGENT", agent_text, "troubleshoot_all")
                gather.append(create_ssml_say(agent_text))
                response.redirect(continue_url)
            else:
                # No troubleshooting steps available — offer image upload or scheduling
                state["step"] = "offer_image_upload"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"I don't have specific troubleshooting steps for that issue{name_phrase}. "
                    "I can send you a link to upload a photo for AI diagnosis, "
                    "or I can schedule a technician. Which would you prefer?"
                ))
                response.redirect(continue_url)
    
    elif current_step == "ask_symptoms":
        state["symptoms"] = speech_result

        # Use LLM to classify what the customer said — including schedule/callback redirects
        symptom_intent = llm_classify_user_intent(
            speech_result,
            choices=["describe_problem", "unsure", "schedule", "callback"],
            context="Agent asked the customer to describe what's wrong with their appliance. "
                    "Customer may describe the problem, say they're unsure, ask to schedule a technician, or want to call back."
        )
        intent_choice = symptom_intent.get("choice", "unclear")
        intent_conf = symptom_intent.get("confidence", 0.0)

        if intent_choice == "schedule" and intent_conf >= 0.6:
            state["step"] = "collect_zip"
            if not state.get("symptom_summary"):
                appliance = state.get('appliance_type', 'appliance')
                state["symptom_summary"] = f"Your {appliance} is not working properly"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Absolutely{name_phrase}! Let's get a technician scheduled. "
                "What is your ZIP code?"
            ))
            response.redirect(continue_url)
            return Response(content=str(response), media_type="application/xml")

        if intent_choice == "callback" and intent_conf >= 0.6:
            state["step"] = "done"
            update_state(call_sid, state)
            
            goodbye_text = (
                f"No problem{name_phrase}. You can call us back anytime when you're ready. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            say_obj = say_with_logging(goodbye_text, call_sid, "ask_symptoms")
            response.append(say_obj)
            response.hangup()
            return Response(content=str(response), media_type="application/xml")

        if intent_choice == "unsure" and intent_conf >= 0.6:
            appliance = state.get('appliance_type', 'appliance')
            state["symptom_summary"] = (
                f"Your {appliance} is not working properly"
            )
            state["error_codes"] = []
            state["is_urgent"] = False
            state["step"] = "offer_troubleshoot_or_schedule"
            update_state(call_sid, state)

            logger.info(f"Symptoms captured (unsure): {speech_result[:100]}", extra={"call_sid": call_sid, "step": "ask_symptoms"})

            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"No worries{name_phrase}. "
                "Would you like me to walk you through a few quick troubleshooting steps, "
                "or would you prefer to schedule a technician right away?"
            ))
            response.redirect(continue_url)
        elif intent_choice == "describe_problem" and intent_conf >= 0.6:
            extracted = llm_extract_symptoms(speech_result)
            summary = extracted.get("symptom_summary") or speech_result
            # Avoid speaking awkward meta-text back to the customer.
            summary_lower = summary.lower()
            meta_patterns = [
                "the caller", "the customer", "the user",
                "customer reported", "caller described", "user said",
                "customer's ", "caller's ", "user's ",
                "no error codes", "no specific", "no further",
                "reported that", "describes a", "mentioned that",
            ]
            if any(p in summary_lower for p in meta_patterns):
                appliance = state.get('appliance_type', 'appliance')
                summary = f"Your {appliance} is not working properly"
            state["symptom_summary"] = summary
            state["error_codes"] = extracted.get("error_codes") or []
            state["is_urgent"] = bool(extracted.get("is_urgent"))

            state["step"] = "offer_troubleshoot_or_schedule"
            update_state(call_sid, state)

            logger.info(f"Symptoms captured: {speech_result[:100]}", extra={"call_sid": call_sid, "step": "ask_symptoms"})

            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"I understand{name_phrase}. {summary}. "
                "Would you like me to walk you through a few quick troubleshooting steps, "
                "or would you prefer to schedule a technician right away?"
            ))
            response.redirect(continue_url)
        else:
            # Unclear or low-detail — ask for more specifics
            update_state(call_sid, state)
            gather = _build_gather(response, continue_url, timeout=10, speech_timeout="4")
            gather.append(create_ssml_say(
                f"Thanks{name_phrase}. I heard it's your {state.get('appliance_type', 'appliance')}. "
                "What exactly is the issue? For example, not cooling, leaking, making noise, or an error code. "
                "If you don't know, just say I don't know."
            ))
            response.redirect(continue_url)
    
    # ==================== TROUBLESHOOTING (ALL STEPS AT ONCE) ====================
    
    elif current_step == "troubleshoot_all":
        # Check if response is too short/garbled — likely captured while agent
        # was still speaking the troubleshooting steps.
        clean_text = re.sub(r'[^a-zA-Z\s]', '', speech_result).strip()
        ts_attempts = state.get("troubleshoot_reprompt", 0)
        
        if len(clean_text) < 10 and ts_attempts < 2:
            state["troubleshoot_reprompt"] = ts_attempts + 1
            update_state(call_sid, state)
            
            logger.debug(f"Troubleshoot response too short ({len(clean_text)} chars), re-prompting", extra={"call_sid": call_sid, "step": "troubleshoot_all"})
            
            gather = _build_gather(response, continue_url, timeout=15, speech_timeout="4")
            gather.append(create_ssml_say(
                f"Take your time{name_phrase}. Once you've tried those steps, "
                "let me know if any of them helped, or if you'd like to schedule a technician."
            ))
            response.redirect(continue_url)
        else:
            state["troubleshoot_reprompt"] = 0
            
            # Use LLM to interpret the customer's response to troubleshooting
            ts_steps_text = state.get("troubleshooting_steps_text", "")
            interpretation = llm_interpret_troubleshooting_response(speech_result, ts_steps_text)
            logger.debug(f"Troubleshoot interpretation: {interpretation}", extra={"call_sid": call_sid})
            
            # ONLY treat as resolved if customer EXPLICITLY confirmed the fix worked
            # with HIGH confidence. "I checked it" or "I tried that" is NOT resolved.
            explicitly_resolved = (
                interpretation.get("is_resolved") is True
                and interpretation.get("confidence") == "high"
            )
            
            if explicitly_resolved:
                state["resolved"] = True
                state["step"] = "done"
                update_state(call_sid, state)
                
                log_call_end(call_sid, resolved=True, reason="Troubleshooting successful")
                
                agent_text = (
                    f"Wonderful{name_phrase}! I'm so glad that helped! "
                    "If you ever have any other issues, don't hesitate to give us a call. "
                    "Have a great day, and thank you for choosing Sears Home Services. Take care!"
                )
                log_conversation(call_sid, "AGENT", agent_text, "troubleshoot_all")
                response.append(create_ssml_say(agent_text))
                response.hangup()
                return Response(content=str(response), media_type="application/xml")
            
            # Customer did NOT explicitly confirm resolution — classify next action
            # Use LLM to determine what the customer wants to do next
            next_intent = llm_classify_user_intent(
                speech_result,
                choices=["schedule", "photo", "resolved", "not_resolved"],
                context="Customer tried troubleshooting steps and reported the result. "
                        "They did NOT say the problem is fixed. What do they want to do next?"
            )
            next_choice = next_intent.get("choice", "unclear")
            
            logger.debug(f"Troubleshoot next intent: {next_intent}", extra={"call_sid": call_sid, "step": "troubleshoot_all"})
            
            if next_choice == "schedule":
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"No problem{name_phrase}! Let's get a technician scheduled. "
                    "What is your ZIP code?"
                ))
                response.redirect(continue_url)
            
            elif next_choice == "photo":
                state["step"] = "collect_email"
                state["email_attempts"] = 0
                update_state(call_sid, state)
                
                logger.info("User chose image upload from troubleshoot_all", extra={"call_sid": call_sid, "step": "troubleshoot_all"})
                
                email_hints = ("gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                              "at gmail dot com, dot com, dot net, at the rate, "
                              "zero, one, two, three, four, five, six, seven, eight, nine")
                gather = _build_gather(response, continue_url, timeout=15, speech_timeout="5",
                                       hints=email_hints, language="en-US")
                gather.append(create_ssml_say(
                    f"Sure{name_phrase}! I'll send you a link to upload a photo. "
                    "What's your email address? You can spell it out letter by letter if that's easier."
                ))
                response.redirect(continue_url)
            
            elif next_choice == "resolved":
                state["step"] = "confirm_resolution"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"That's great{name_phrase}! So the issue is resolved? "
                    "Just say yes to confirm, or no if you still need help."
                ))
                response.redirect(continue_url)
            
            else:
                # Not resolved or unclear — offer image upload or scheduling
                state["step"] = "offer_image_upload"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"I understand{name_phrase}, those steps didn't help. "
                    "I can send you a link to upload a photo of your appliance for AI diagnosis, "
                    "or I can help you schedule a technician to come take a look. "
                    "Which would you prefer?"
                ))
                response.redirect(continue_url)
    
    elif current_step == "confirm_resolution":
        # 100% LLM-powered yes/no classification
        llm_result = llm_classify_yes_no(
            speech_result,
            context="Agent asked: Is the issue resolved?"
        )
        intent = llm_result.get("intent", "unclear")
        
        logger.debug(f"Resolution confirm LLM: {llm_result}", extra={"call_sid": call_sid, "step": "confirm_resolution"})
        
        if intent == "yes":
            state["resolved"] = True
            state["step"] = "done"
            update_state(call_sid, state)
            
            log_call_end(call_sid, resolved=True, reason="Troubleshooting successful")
            
            response.append(create_ssml_say(
                f"Wonderful{name_phrase}! I'm so glad that worked! "
                "If you ever have any other issues, don't hesitate to give us a call. "
                "Have a great day, and thank you for choosing Sears Home Services!"
            ))
            response.hangup()
        elif intent == "no" or intent == "unclear":
            state["step"] = "offer_image_upload"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"No worries{name_phrase}. I can send you a link to upload a photo for AI diagnosis, "
                "or schedule a technician. Which would you prefer?"
            ))
            response.redirect(continue_url)
    
    elif current_step == "offer_image_upload":
        # 100% LLM-powered intent classification
        llm_result = llm_classify_user_intent(
            speech_result,
            choices=["photo", "schedule", "callback"],
            context="Agent asked: Would you like to upload a photo for AI diagnosis, or schedule a technician?"
        )
        choice = llm_result.get("choice", "unclear")
        conf = llm_result.get("confidence", 0.0)
        
        logger.debug(f"Image/schedule LLM: {llm_result}", extra={"call_sid": call_sid, "step": "offer_image_upload"})
        
        if choice == "unclear" or conf < 0.5:
            choice = None  # Ask again
        
        # Execute the choice
        if choice == "photo":
            state["step"] = "collect_email"
            state["email_attempts"] = 0
            update_state(call_sid, state)
            
            logger.info("User chose image upload (Tier 3)", extra={"call_sid": call_sid, "step": "offer_image_upload"})
            
            email_hints = ("gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                          "at gmail dot com, dot com, dot net, at the rate, "
                          "zero, one, two, three, four, five, six, seven, eight, nine")
            gather = _build_gather(response, continue_url, timeout=15, speech_timeout="5",
                                   hints=email_hints, language="en-US")
            gather.append(create_ssml_say(
                f"Perfect{name_phrase}! I'll send you a link to upload your photo. "
                "What's your email address? You can spell it out letter by letter if that's easier."
            ))
            response.redirect(continue_url)
        
        elif choice == "schedule":
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            logger.info("User chose technician scheduling", extra={"call_sid": call_sid, "step": "offer_image_upload"})
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Absolutely{name_phrase}! Let me help you schedule a technician visit. "
                "What is your ZIP code?"
            ))
            response.redirect(continue_url)

        elif choice == "callback":
            state["step"] = "done"
            update_state(call_sid, state)

            logger.info("User deferred service and will call back", extra={"call_sid": call_sid, "step": "offer_image_upload"})

            goodbye_text = (
                f"No problem{name_phrase}. You can call us back anytime when you're ready. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            say_obj = say_with_logging(goodbye_text, call_sid, "offer_image_upload")
            response.append(say_obj)
            response.hangup()
        
        else:
            # Still unclear — ask again
            gather = _build_gather(response, continue_url, timeout=10, speech_timeout="5")
            gather.append(create_ssml_say(
                "I'm sorry, I didn't catch that. "
                "Would you like to upload a photo for diagnosis, "
                "or schedule a technician visit?"
            ))
            response.redirect(continue_url)
    
    # =========================================================================
    # EMAIL CAPTURE with natural readback + spelling
    # =========================================================================
    
    elif current_step == "collect_email":
        # Cross-cutting: detect if customer wants to change course instead of giving email
        redirect_intent = llm_classify_user_intent(
            speech_result,
            choices=["email", "schedule", "callback"],
            context="Agent asked for the customer's email address to send a photo upload link. "
                    "Did the customer provide an email, or do they want to schedule a technician or call back instead?"
        )
        redirect_choice = redirect_intent.get("choice", "email")
        redirect_conf = redirect_intent.get("confidence", 0.0)
        
        if redirect_choice == "schedule" and redirect_conf >= 0.6:
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"Sure{name_phrase}! Let's schedule a technician instead. "
                "What is your ZIP code?"
            ))
            response.redirect(continue_url)
            return Response(content=str(response), media_type="application/xml")
        
        if redirect_choice == "callback" and redirect_conf >= 0.6:
            state["step"] = "done"
            update_state(call_sid, state)
            goodbye_text = (
                f"No problem{name_phrase}. You can call us back anytime when you're ready. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            say_obj = say_with_logging(goodbye_text, call_sid, "collect_email")
            response.append(say_obj)
            response.hangup()
            return Response(content=str(response), media_type="application/xml")
        
        email = extract_email_from_speech(speech_result, call_sid)
        
        state["pending_email"] = email
        state["step"] = "confirm_email"
        if "email_confirm_attempts" not in state:
            state["email_confirm_attempts"] = 0
        update_state(call_sid, state)
        
        logger.info(f"Email captured: {email}, awaiting confirmation", extra={"call_sid": call_sid, "step": "collect_email"})
        
        # Use natural readback first, then spelling (per recruiter feedback)
        email_readback = speak_email_naturally(email)
        
        gather = _build_gather(response, continue_url, timeout=7, speech_timeout="3", language="en-US")
        gather.append(create_ssml_say(email_readback))
        response.redirect(continue_url)
    
    elif current_step == "confirm_email":
        pending_email = state.get("pending_email")
        
        # 100% LLM-powered yes/no/correction classification
        llm_result = llm_classify_yes_no(
            speech_result,
            context=f"Agent asked: Is email {pending_email} correct?"
        )
        intent = llm_result.get("intent", "unclear")
        correction = llm_result.get("correction_value")
        
        logger.debug(f"Email confirm LLM: {llm_result}", extra={"call_sid": call_sid, "step": "confirm_email"})
        
        if intent == "yes":
            state["customer_email"] = pending_email
            state["pending_email"] = None
            
            logger.info(f"Email confirmed: {pending_email}", extra={"call_sid": call_sid, "step": "confirm_email"})
            
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
                
                logger.info("Upload link sent, entering wait loop", extra={"call_sid": call_sid, "step": "confirm_email"})
                
                gather = _build_gather(response, continue_url, timeout=15, speech_timeout="3")
                gather.append(create_ssml_say(
                    "I've sent an upload link to your email. "
                    "Please check your inbox, click the link, and upload a clear photo of your appliance. "
                    "I'll stay on the line while you do this. "
                    "Once you've uploaded the image, just say done or uploaded. "
                    "If you'd rather skip and schedule a technician, say skip."
                ))
                response.redirect(continue_url)
                
            except Exception as e:
                log_error(call_sid, e, step="confirm_email", context="Error creating upload token")
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
                gather.append(create_ssml_say(
                    "I'm sorry, there was an issue sending the upload link. "
                    "Let me help you schedule a technician instead. "
                    "What is your ZIP code?"
                ))
                response.redirect(continue_url)
        
        elif intent == "no" or intent == "correction":
            state["pending_email"] = None
            
            # Check if customer wants to change course (schedule/callback) instead of re-trying email
            redirect_intent = llm_classify_user_intent(
                speech_result,
                choices=["retry_email", "schedule", "callback"],
                context="Customer said NO to email confirmation. Are they just correcting the email "
                        "(retry_email), or do they want to schedule a technician or call back instead?"
            )
            redirect_choice = redirect_intent.get("choice", "retry_email")
            redirect_conf = redirect_intent.get("confidence", 0.0)
            
            logger.debug(f"Email reject redirect: {redirect_intent}", extra={"call_sid": call_sid, "step": "confirm_email"})
            
            if redirect_choice == "schedule" and redirect_conf >= 0.5:
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"Sure{name_phrase}! Let's schedule a technician instead. "
                    "What is your ZIP code?"
                ))
                response.redirect(continue_url)
            
            elif redirect_choice == "callback" and redirect_conf >= 0.5:
                state["step"] = "done"
                update_state(call_sid, state)
                goodbye_text = (
                    f"No problem{name_phrase}. You can call us back anytime when you're ready. "
                    "Thank you for calling Sears Home Services. Goodbye."
                )
                say_obj = say_with_logging(goodbye_text, call_sid, "confirm_email")
                response.append(say_obj)
                response.hangup()
            
            else:
                # Customer just wants to correct the email — retry
                state["email_confirm_attempts"] = state.get("email_confirm_attempts", 0) + 1
                logger.debug(f"Email rejected, attempt {state['email_confirm_attempts']}", extra={"call_sid": call_sid, "step": "confirm_email"})
                
                if state["email_confirm_attempts"] <= 2:
                    state["step"] = "collect_email"
                    update_state(call_sid, state)
                    
                    email_hints = ("gmail.com, yahoo.com, outlook.com, hotmail.com, icloud.com, "
                                  "at gmail dot com, dot com, dot net, at the rate, "
                                  "zero, one, two, three, four, five, six, seven, eight, nine")
                    gather = _build_gather(response, continue_url, timeout=10, speech_timeout="4",
                                           hints=email_hints, language="en-US")
                    gather.append(create_ssml_say(
                        "No problem, let's try again. "
                        "Please spell your email slowly, letter by letter. "
                        "Say dot for periods and at for the at symbol."
                    ))
                    response.redirect(continue_url)
                else:
                    logger.warning("Email confirmation failed 3 times, falling back to scheduling", extra={"call_sid": call_sid, "step": "confirm_email"})
                    state["step"] = "collect_zip"
                    update_state(call_sid, state)
                    
                    gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3", language="en-US")
                    gather.append(create_ssml_say(
                        "I'm having trouble with the email. "
                        "Let me help you schedule a technician instead. "
                        "What is your ZIP code?"
                    ))
                    response.redirect(continue_url)
        
        else:
            # Still unclear — ask again
            gather = _build_gather(response, continue_url, timeout=7, speech_timeout="3", language="en-US")
            spelled = _spell_email_slow(pending_email) if pending_email else "the email"
            gather.append(create_ssml_say(
                f"I need a yes or no. Is {spelled} correct?"
            ))
            response.redirect(continue_url)
    
    # =========================================================================
    # IMAGE UPLOAD WAITING + ANALYSIS
    # =========================================================================
    
    elif current_step == "waiting_for_upload":
        # Always check if image was uploaded automatically first
        upload_status = get_upload_status_by_call_sid(call_sid)
        if upload_status and upload_status.get("analysis_ready"):
            state["step"] = "speak_analysis"
            state["waiting_for_upload"] = False
            state["upload_wait_attempts"] = 0
            update_state(call_sid, state)
            logger.info("Auto-detected image upload during speech", extra={"call_sid": call_sid, "step": "waiting_for_upload"})
            response.redirect(continue_url)
            return Response(content=str(response), media_type="application/xml")
        
        # 100% LLM-powered intent classification for upload waiting
        upload_intent = llm_interpret_upload_intent(speech_result)
        logger.debug(f"Upload intent LLM: {upload_intent}", extra={"call_sid": call_sid, "step": "waiting_for_upload"})
        
        if upload_intent == "resend":
            upload_url = reset_upload_for_reupload(call_sid)
            if upload_url:
                email = state.get("customer_email", "")
                if email:
                    send_upload_email(email, upload_url, state.get("appliance_type"))
                    logger.info(f"Re-sent upload email to {email}", extra={"call_sid": call_sid, "step": "waiting_for_upload"})
                
                gather = _build_gather(response, continue_url, timeout=30, speech_timeout="3")
                agent_text = "I've re-sent the upload link to your email. Please upload a new photo and say done when you're finished, or say skip to schedule a technician."
                log_conversation(call_sid, "AGENT", agent_text, "waiting_for_upload")
                gather.append(create_ssml_say(agent_text))
                response.redirect(continue_url)
                return Response(content=str(response), media_type="application/xml")
        
        if upload_intent == "more_time":
            state["upload_wait_attempts"] = 0
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=30, speech_timeout="3")
            agent_text = "No problem, take your time. Just let me know when you've uploaded the image, or say skip to schedule a technician."
            log_conversation(call_sid, "AGENT", agent_text, "waiting_for_upload")
            gather.append(create_ssml_say(agent_text))
            response.redirect(continue_url)
            return Response(content=str(response), media_type="application/xml")
        
        if upload_intent == "done":
            upload_status = get_upload_status_by_call_sid(call_sid)
            
            if upload_status and upload_status.get("analysis_ready"):
                state["step"] = "speak_analysis"
                state["waiting_for_upload"] = False
                update_state(call_sid, state)
                response.redirect(continue_url)
            
            elif upload_status and upload_status.get("image_uploaded"):
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=10, speech_timeout="3")
                gather.append(create_ssml_say(
                    "I see your image was received. Just a moment while I analyze it. "
                    "Say ready when you'd like me to check again."
                ))
                response.pause(length=5)
                response.redirect(continue_url)
            
            else:
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                if state["upload_poll_count"] < MAX_UPLOAD_POLL_COUNT:
                    gather = _build_gather(response, continue_url, timeout=15, speech_timeout="3")
                    gather.append(create_ssml_say(
                        "I don't see the upload yet. Please check your email for the link. "
                        "Let me know when you've uploaded the image, or say skip to continue without it."
                    ))
                    response.redirect(continue_url)
                else:
                    state["step"] = "collect_zip"
                    state["waiting_for_upload"] = False
                    update_state(call_sid, state)
                    
                    gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
                    gather.append(create_ssml_say(
                        "We've been waiting a while. Let's continue with scheduling a technician. "
                        "You can still upload the photo later using the link in your email. "
                        "What is your ZIP code?"
                    ))
                    response.redirect(continue_url)
        
        elif upload_intent == "skip":
            state["step"] = "collect_zip"
            state["waiting_for_upload"] = False
            update_state(call_sid, state)
            
            logger.info("User skipped upload, moving to scheduling", extra={"call_sid": call_sid, "step": "waiting_for_upload"})
            
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(
                "No problem. You can still upload the photo later using the email link. "
                "Let's schedule a technician. What is your ZIP code?"
            ))
            response.redirect(continue_url)
        
        else:
            # Unclear — check upload status and re-prompt
            upload_status = get_upload_status_by_call_sid(call_sid)
            
            if upload_status and upload_status.get("analysis_ready"):
                state["step"] = "speak_analysis"
                state["waiting_for_upload"] = False
                update_state(call_sid, state)
                response.redirect(continue_url)
            else:
                state["upload_poll_count"] = state.get("upload_poll_count", 0) + 1
                update_state(call_sid, state)
                
                if state["upload_poll_count"] < MAX_UPLOAD_POLL_COUNT:
                    gather = _build_gather(response, continue_url, timeout=15, speech_timeout="3")
                    gather.append(create_ssml_say(
                        "I'm still here. Let me know once you've uploaded the image, "
                        "or say skip to schedule a technician instead."
                    ))
                    response.redirect(continue_url)
                else:
                    state["step"] = "collect_zip"
                    state["waiting_for_upload"] = False
                    update_state(call_sid, state)
                    
                    gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
                    gather.append(create_ssml_say(
                        "We've been waiting a while. Let's continue with scheduling. "
                        "What is your ZIP code?"
                    ))
                    response.redirect(continue_url)
    
    elif current_step == "speak_analysis":
        upload_status = get_upload_status_by_call_sid(call_sid)
        
        if state.get("analysis_spoken"):
            state["step"] = "after_analysis"
            state["waiting_for_upload"] = False
            update_state(call_sid, state)
            response.redirect(continue_url)
            return Response(content=str(response), media_type="application/xml")
        
        if not upload_status or not upload_status.get("analysis_ready"):
            state["step"] = "collect_zip"
            state["waiting_for_upload"] = False
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(
                "I'm sorry, the image analysis isn't available yet. "
                "Let's schedule a technician to take a look. What is your ZIP code?"
            ))
            response.redirect(continue_url)
        
        elif upload_status.get("is_appliance_image") == False:
            appliance = state.get("appliance_type") or "appliance"
            state["step"] = "waiting_for_upload"
            state["waiting_for_upload"] = True
            state["upload_poll_count"] = 0
            update_state(call_sid, state)
            
            logger.info("Image was not an appliance, asking for re-upload", extra={"call_sid": call_sid, "step": "speak_analysis"})
            
            # Reset the upload token so the old analysis doesn't trigger auto-detect loop
            upload_url = reset_upload_for_reupload(call_sid)
            reupload_msg = ""
            if upload_url:
                reupload_msg = " I've re-sent the upload link to your email. "
            
            gather = _build_gather(response, continue_url, timeout=30, speech_timeout="3")
            gather.append(create_ssml_say(
                f"The image doesn't appear to show the {appliance}. "
                "Please upload a clear photo of the appliance itself, "
                "especially showing any error codes or the problem area."
                f"{reupload_msg}"
                "Say done when you've uploaded a new photo, or skip to schedule a technician."
            ))
            response.redirect(continue_url)
        
        else:
            summary = upload_status.get("analysis_summary", "")
            tips = upload_status.get("troubleshooting_tips", "")
            
            state["analysis_spoken"] = True
            state["step"] = "after_analysis"
            state["waiting_for_upload"] = False
            update_state(call_sid, state)
            
            logger.info("Speaking analysis results to user", extra={"call_sid": call_sid, "step": "speak_analysis"})
            
            analysis_speech = "I've analyzed your image. "
            
            # Keep summary brief for phone — just the key finding
            if summary:
                # Truncate at sentence boundary if possible
                if len(summary) > 150:
                    dot_pos = summary[:150].rfind(".")
                    summary = summary[:dot_pos + 1] if dot_pos > 50 else summary[:147] + "..."
                analysis_speech += summary + " "
            
            # Keep tips very short — just the most actionable one
            if tips:
                # Extract just the first tip/step if multiple
                first_tip = tips
                for sep in ["Step 2:", "2.", "2)", "\n"]:
                    idx = tips.find(sep)
                    if idx > 10:
                        first_tip = tips[:idx].strip()
                        break
                if len(first_tip) > 120:
                    dot_pos = first_tip[:120].rfind(".")
                    first_tip = first_tip[:dot_pos + 1] if dot_pos > 30 else first_tip[:117] + "..."
                analysis_speech += f"Quick tip: {first_tip} "
            
            analysis_speech += "Would you like to try that, or should I schedule a technician?"
            
            gather = _build_gather(response, continue_url, timeout=10, speech_timeout="3")
            log_conversation(call_sid, "AGENT", analysis_speech, "speak_analysis")
            gather.append(create_ssml_say(analysis_speech))
            response.redirect(continue_url)
    
    elif current_step == "after_analysis":
        # 100% LLM-powered intent classification
        intent = llm_interpret_after_analysis(speech_result)
        logger.debug(f"After-analysis LLM: {intent}", extra={"call_sid": call_sid, "step": "after_analysis"})
        
        if intent == "schedule":
            logger.info("Troubleshooting didn't help, offering technician", extra={"call_sid": call_sid, "step": "after_analysis"})
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
            gather.append(create_ssml_say(
                f"I'm sorry the troubleshooting didn't resolve the issue{name_phrase}. "
                "Let me schedule a technician for you. What is your ZIP code?"
            ))
            response.redirect(continue_url)
        
        elif intent == "resolved":
            state["resolved"] = True
            state["step"] = "done"
            update_state(call_sid, state)
            
            log_call_end(call_sid, resolved=True, reason="Issue resolved after image analysis")
            
            response.append(create_ssml_say(
                "Great, I'm glad that helped! "
                "If the issue comes back, you can always call us again. "
                "Thank you for calling Sears Home Services. Goodbye."
            ))
            response.hangup()
        
        elif intent == "try_fix":
            state["step"] = "done"
            state["resolved"] = True
            update_state(call_sid, state)
            
            log_call_end(call_sid, resolved=True, reason="Customer will try suggested fix")
            
            response.append(create_ssml_say(
                f"Sounds good{name_phrase}! Give that a try. "
                "If the issue persists, you can always call us back and we'll schedule a technician. "
                "Thank you for calling Sears Home Services. Good luck!"
            ))
            response.hangup()
        
        else:
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(
                "Would you like to try the suggested fix, or would you prefer to schedule a technician?"
            ))
            response.redirect(continue_url)
    
    # =========================================================================
    # SCHEDULING: ZIP → Confirm ZIP → Time Pref → Slots → Book
    # =========================================================================
    
    elif current_step == "collect_zip":
        # Use LLM-powered ZIP extraction (handles number words, STT artifacts)
        zip_code = llm_extract_zip_code(speech_result)
        
        if zip_code:
            state["zip_code"] = zip_code
            state["zip_attempts"] = 0
            state["step"] = "confirm_zip"
            update_state(call_sid, state)
            
            logger.info(f"ZIP code captured: {zip_code}", extra={"call_sid": call_sid, "step": "collect_zip"})
            
            zip_confirm_text = (
                f"I heard ZIP code {' '.join(zip_code)}. Is that correct?"
            )
            log_conversation(call_sid, "AGENT", zip_confirm_text, "collect_zip")
            
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(zip_confirm_text))
            response.redirect(continue_url)
        else:
            state["zip_attempts"] = state.get("zip_attempts", 0) + 1
            update_state(call_sid, state)
            
            logger.debug(f"ZIP attempt {state['zip_attempts']}/3, input: '{speech_result}'", extra={"call_sid": call_sid, "step": "collect_zip"})
            
            if state["zip_attempts"] < 3:
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
                gather.append(create_ssml_say(
                    "I'm sorry, I didn't catch a valid ZIP code. "
                    "Please say your 5-digit ZIP code clearly, like 6 0 6 0 1."
                ))
                response.redirect(continue_url)
            else:
                state["step"] = "done"
                update_state(call_sid, state)
                
                logger.warning("ZIP capture failed after 3 attempts", extra={"call_sid": call_sid, "step": "collect_zip"})
                
                response.append(create_ssml_say(
                    "I'm having trouble understanding the ZIP code. "
                    "Please visit our website or call back to schedule your appointment. "
                    "Thank you for calling Sears Home Services. Goodbye."
                ))
                response.hangup()
    
    elif current_step == "confirm_zip":
        zip_code = state.get("zip_code", "")
        
        # 100% LLM-powered yes/no/correction classification
        llm_result = llm_classify_yes_no(
            speech_result,
            context=f"Agent asked: Is ZIP code {zip_code} correct?"
        )
        intent = llm_result.get("intent", "unclear")
        correction = llm_result.get("correction_value")
        
        logger.debug(f"ZIP confirm LLM: {llm_result}", extra={"call_sid": call_sid, "step": "confirm_zip"})
        
        if intent == "yes":
            state["step"] = "collect_time_pref"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(
                "Do you prefer a morning or afternoon appointment?"
            ))
            response.redirect(continue_url)
        elif intent == "no":
            state["zip_code"] = None
            state["step"] = "collect_zip"
            update_state(call_sid, state)
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
            gather.append(create_ssml_say(
                "No problem, let me get that again. What is your ZIP code?"
            ))
            response.redirect(continue_url)
        elif intent == "correction" and correction:
            # User provided corrected ZIP inline (e.g., "no it's 60604")
            corrected_zip = llm_extract_zip_code(correction)
            if corrected_zip:
                state["zip_code"] = corrected_zip
                state["step"] = "confirm_zip"
                update_state(call_sid, state)
                
                logger.info(f"ZIP corrected to: {corrected_zip}", extra={"call_sid": call_sid, "step": "confirm_zip"})
                
                gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
                gather.append(create_ssml_say(
                    f"Got it. So that's {' '.join(corrected_zip)}. Is that correct?"
                ))
                response.redirect(continue_url)
            else:
                state["zip_code"] = None
                state["step"] = "collect_zip"
                update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="4")
                gather.append(create_ssml_say(
                    "I didn't catch that. What is your correct ZIP code?"
                ))
                response.redirect(continue_url)
        else:
            # Still unclear — ask again
            gather = _build_gather(response, continue_url, timeout=5, speech_timeout="3")
            gather.append(create_ssml_say(
                f"I need a yes or no. Is {' '.join(zip_code)} your correct ZIP code?"
            ))
            response.redirect(continue_url)
    
    elif current_step == "collect_time_pref":
        # 100% LLM-powered time preference extraction
        time_pref = llm_extract_time_preference(speech_result)
        
        state["time_preference"] = time_pref
        
        logger.info(f"Time preference: {time_pref}", extra={"call_sid": call_sid, "step": "collect_time_pref"})
        
        slots = find_available_slots(
            zip_code=state.get("zip_code"),
            appliance_type=state.get("appliance_type"),
            time_preference=time_pref,
            limit=3
        )
        
        if not slots:
            state["step"] = "done"
            update_state(call_sid, state)
            
            no_slots_text = (
                "I'm sorry, we don't have any technicians available in your area "
                f"for {state.get('appliance_type')} service at this time. "
                "Please call back later or visit our website to schedule. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            say_obj = say_with_logging(no_slots_text, call_sid, "collect_time_pref")
            response.append(say_obj)
            response.hangup()
        else:
            state["offered_slots"] = slots
            state["step"] = "choose_slot"
            update_state(call_sid, state)
            
            slot_speech = "Here are the available appointments: "
            for i, slot in enumerate(slots, 1):
                slot_speech += format_slot_for_speech(slot, i) + ". "
            
            slot_options_text = slot_speech + "Please say option 1, option 2, or option 3 to select your preferred time."
            log_conversation(call_sid, "AGENT", slot_options_text, "collect_time_pref")
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(slot_options_text))
            response.redirect(continue_url)
    
    elif current_step == "choose_slot":
        offered_slots = state.get("offered_slots", [])
        
        logger.debug(f"Offered slots count: {len(offered_slots)}, User said: '{speech_result}'", extra={"call_sid": call_sid, "step": "choose_slot"})
        
        # First check for escape intents (troubleshoot / cancel) via LLM
        escape_result = llm_classify_user_intent(
            speech_result,
            choices=["select_slot", "troubleshoot", "cancel"],
            context="Agent offered 3 appointment slots. Customer should pick one, or they might want troubleshooting or to cancel."
        )
        escape_choice = escape_result.get("choice", "unclear")
        escape_conf = escape_result.get("confidence", 0.0)
        
        logger.debug(f"Slot escape LLM: {escape_result}", extra={"call_sid": call_sid, "step": "choose_slot"})
        
        if escape_choice == "troubleshoot" and escape_conf >= 0.6:
            state["step"] = "offer_troubleshoot_or_schedule"
            update_state(call_sid, state)
            
            appliance = state.get("appliance_type", "appliance")
            logger.info("Customer wants troubleshooting instead of scheduling", extra={"call_sid": call_sid, "step": "choose_slot"})
            
            gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
            gather.append(create_ssml_say(
                f"No problem{name_phrase}! Would you like me to walk you through some troubleshooting steps for your {appliance}?"
            ))
            response.redirect(continue_url)
        
        elif escape_choice == "cancel" and escape_conf >= 0.6:
            state["step"] = "done"
            update_state(call_sid, state)
            
            logger.info("Customer cancelled from slot selection", extra={"call_sid": call_sid, "step": "choose_slot"})
            
            goodbye_text = (
                f"No problem{name_phrase}. You can call us back anytime to schedule. "
                "Thank you for calling Sears Home Services. Goodbye!"
            )
            say_obj = say_with_logging(goodbye_text, call_sid, "choose_slot")
            response.append(say_obj)
            response.hangup()
        
        else:
            # Use LLM to match slot selection from natural speech
            # Build a description of offered slots for the LLM
            slots_desc = ""
            for i, slot in enumerate(offered_slots):
                start = slot.get("start_time")
                if start:
                    day_name = start.strftime("%A")
                    date_str = start.strftime("%B %d")
                    hour = start.hour
                    if hour < 12:
                        time_str = f"{hour} AM" if hour > 0 else "12 AM"
                    else:
                        hour_12 = hour - 12 if hour > 12 else hour
                        time_str = f"{hour_12} PM" if hour_12 > 0 else "12 PM"
                    slots_desc += f"Option {i+1} (index {i}): {day_name}, {date_str} at {time_str}\n"
            
            chosen_index = llm_choose_slot(speech_result, slots_desc) if slots_desc else None
            
            logger.debug(f"Chosen index (LLM): {chosen_index}, Slots available: {len(offered_slots)}", extra={"call_sid": call_sid, "step": "choose_slot"})
            
            if chosen_index is not None and chosen_index < len(offered_slots) and len(offered_slots) > 0:
                chosen_slot = offered_slots[chosen_index]
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
                    
                    logger.info(f"Appointment booked: ID={appt_info['id']}", extra={"call_sid": call_sid, "step": "choose_slot"})
                    
                    start = appt_info["start_time"]
                    day_name = start.strftime("%A")
                    date_str = start.strftime("%B %d")
                    hour = start.hour
                    if hour < 12:
                        time_str = f"{hour} AM" if hour > 0 else "12 AM"
                    else:
                        hour_12 = hour - 12 if hour > 12 else hour
                        time_str = f"{hour_12} PM" if hour_12 > 0 else "12 PM"
                    
                    confirmation_text = (
                        f"Your appointment is confirmed for {day_name}, {date_str} at {time_str} "
                        f"with technician {appt_info['technician_name']}. "
                        "You will receive a confirmation text shortly. "
                        "Thank you for calling Sears Home Services. Goodbye."
                    )
                    say_obj = say_with_logging(confirmation_text, call_sid, "choose_slot")
                    response.append(say_obj)
                    response.hangup()
                    
                except Exception as e:
                    log_error(call_sid, e, step="choose_slot", context="Booking failed")
                    state["step"] = "done"
                    update_state(call_sid, state)
                    
                    error_text = (
                        "I'm sorry, there was an error booking your appointment. "
                        "Please call back or visit our website to schedule. "
                        "Thank you for calling Sears Home Services. Goodbye."
                    )
                    say_obj = say_with_logging(error_text, call_sid, "choose_slot")
                    response.append(say_obj)
                    response.hangup()
            else:
                if len(offered_slots) == 0:
                    logger.error("No slots available in state!", extra={"call_sid": call_sid, "step": "choose_slot"})
                    slots = find_available_slots(
                        zip_code=state.get("zip_code"),
                        appliance_type=state.get("appliance_type"),
                        time_preference=state.get("time_preference"),
                        limit=3
                    )
                    if slots:
                        state["offered_slots"] = slots
                        update_state(call_sid, state)
                
                gather = _build_gather(response, continue_url, timeout=8, speech_timeout="3")
                retry_text = (
                    "I didn't catch your selection. "
                    "Please say option 1, option 2, or option 3. "
                    "Or say troubleshoot if you'd like to try fixing it yourself, "
                    "or cancel if you'd like to end the call."
                )
                say_obj = say_with_logging(retry_text, call_sid, "choose_slot")
                gather.append(say_obj)
                response.redirect(continue_url)
    
    else:
        response.append(create_ssml_say(
            "I'm sorry, something went wrong. Please call back later. Goodbye."
        ))
        response.hangup()
    
    # Stamp turn start time so the next voice_continue can filter stale Google STT transcripts
    state["_turn_start_ts"] = time.time()
    update_state(call_sid, state)
    
    return Response(content=str(response), media_type="application/xml")
