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
    validate_email
)

router = APIRouter()

# Absolute URL for Twilio webhooks
VOICE_CONTINUE_URL = f"{APP_BASE_URL}/twilio/voice/continue"


def extract_email_from_speech(speech_text: str) -> str:
    """
    Extract email address from speech-to-text result.
    Handles common speech patterns like "john dot smith at gmail dot com"
    """
    text = speech_text.lower().strip()
    
    # Replace common speech patterns
    replacements = [
        (" at ", "@"),
        (" dot ", "."),
        (" underscore ", "_"),
        (" dash ", "-"),
        (" hyphen ", "-"),
        ("at symbol", "@"),
        ("dot com", ".com"),
        ("dot net", ".net"),
        ("dot org", ".org"),
        ("dot edu", ".edu"),
        ("dot co", ".co"),
        ("gmail.com", "gmail.com"),
        ("yahoo.com", "yahoo.com"),
        ("hotmail.com", "hotmail.com"),
        ("outlook.com", "outlook.com"),
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Remove spaces
    text = text.replace(" ", "")
    
    # Try to find an email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    
    if match:
        return match.group(0)
    
    return text


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
    
    elif current_step == "collect_email":
        # Extract email from speech
        email = extract_email_from_speech(speech_result)
        
        if email and validate_email(email):
            state["customer_email"] = email
            state["email_attempts"] = 0
            
            # Create upload token and send email
            upload_token = create_image_upload_token(
                call_sid=call_sid,
                email=email,
                appliance_type=state.get("appliance_type"),
                symptom_summary=state.get("symptom_summary")
            )
            
            upload_url = build_upload_url(upload_token.token)
            send_upload_email(email, upload_url, state.get("appliance_type"))
            
            state["image_upload_sent"] = True
            state["upload_token"] = upload_token.token
            state["step"] = "after_email_sent"
            update_state(call_sid, state)
            
            print(f"[Tier 3] CallSid: {call_sid} - Upload link sent to {email}")
            
            gather = response.gather(
                input="speech",
                timeout=5,
                speech_timeout="3",
                action=VOICE_CONTINUE_URL,
                method="POST"
            )
            gather.say(
                f"I've sent an upload link to {email}. "
                "Please check your inbox and upload a clear photo of your appliance. "
                "Our AI will analyze it and provide additional troubleshooting tips. "
                "Would you also like to schedule a technician visit as a backup, "
                "or is the email enough for now?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            state["email_attempts"] = state.get("email_attempts", 0) + 1
            update_state(call_sid, state)
            
            if state["email_attempts"] <= 2:
                gather = response.gather(
                    input="speech",
                    timeout=5,
                    speech_timeout="3",
                    action=VOICE_CONTINUE_URL,
                    method="POST"
                )
                gather.say(
                    "I'm sorry, I didn't catch a valid email address. "
                    "Please say your email slowly, spelling out any unusual parts. "
                    "For example, john dot smith at gmail dot com."
                )
                response.redirect(VOICE_CONTINUE_URL)
            else:
                # Too many failed attempts, fall back to scheduling
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
                    "Let's schedule a technician visit instead. "
                    "What is your ZIP code?"
                )
                response.redirect(VOICE_CONTINUE_URL)
    
    elif current_step == "after_email_sent":
        text_lower = speech_result.lower()
        
        if "schedule" in text_lower or "technician" in text_lower or "yes" in text_lower or "backup" in text_lower:
            # User wants scheduling as backup
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
                "Let me schedule a technician as well. "
                "What is your ZIP code?"
            )
            response.redirect(VOICE_CONTINUE_URL)
        
        else:
            # User is satisfied with just the email
            state["step"] = "done"
            update_state(call_sid, state)
            
            response.say(
                "You're all set! Check your email for the upload link. "
                "After you upload a photo, our system will analyze it and provide tips. "
                "If you need further assistance, you can always call back. "
                "Thank you for calling Sears Home Services. Goodbye."
            )
            response.hangup()
    
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
