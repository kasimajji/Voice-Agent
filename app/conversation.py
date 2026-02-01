sessions = {}  # {call_id: state_dict}


def get_state(call_id: str) -> dict:
    """Returns existing state or initializes a new one for the call."""
    if call_id in sessions:
        return sessions[call_id]
    
    initial_state = {
        "step": "ask_appliance",
        "appliance_type": None,
        "symptoms": None,
        "symptom_summary": None,
        "error_codes": [],
        "is_urgent": False,
        "troubleshooting_step": 0,
        "resolved": False,
        "zip_code": None,
        "time_preference": None,
        "offered_slots": [],
        "customer_phone": None,
        "appointment_booked": False,
        "appointment_id": None,
        "no_input_attempts": 0,
        "no_match_attempts": 0,
        "customer_email": None,
        "email_attempts": 0,
        "image_upload_sent": False,
        "upload_token": None
    }
    sessions[call_id] = initial_state
    return initial_state


def update_state(call_id: str, new_state: dict) -> None:
    """Updates the state for a given call."""
    sessions[call_id] = new_state


def infer_appliance_type(user_text: str) -> str | None:
    """Infers appliance type from user text using simple keyword matching."""
    text = user_text.lower()
    if "washer" in text or "washing machine" in text:
        return "washer"
    if "dryer" in text:
        return "dryer"
    if "fridge" in text or "refrigerator" in text:
        return "refrigerator"
    if "dishwasher" in text:
        return "dishwasher"
    if "oven" in text or "stove" in text:
        return "oven"
    if "ac" in text or "air conditioner" in text or "hvac" in text:
        return "hvac"
    return None


BASIC_TROUBLESHOOTING = {
    "washer": [
        "Please check that the washer is plugged in and the outlet has power.",
        "Make sure the door is fully closed and latched.",
        "Check your home's breaker panel to see if the washer's breaker is tripped."
    ],
    "refrigerator": [
        "Please check that the fridge is plugged in and the outlet has power.",
        "Make sure the temperature dial is not set to off or the warmest setting.",
        "Check that the doors are closing fully and the seals look intact."
    ],
    "dryer": [
        "Please check that the dryer is plugged in and the outlet has power.",
        "Make sure the lint filter is not completely clogged.",
        "Check your home's breaker panel to see if the dryer's breaker is tripped."
    ],
    "dishwasher": [
        "Please check that the dishwasher is plugged in and the outlet has power.",
        "Make sure the door is fully closed and latched.",
        "Check that the water supply valve is turned on."
    ],
    "oven": [
        "Please check that the oven is plugged in or the gas supply is on.",
        "Make sure the clock is set, as some ovens won't operate without it.",
        "Check your home's breaker panel to see if the oven's breaker is tripped."
    ],
    "hvac": [
        "Please check that the thermostat is set to the correct mode and temperature.",
        "Make sure the air filter is not completely clogged.",
        "Check your home's breaker panel to see if the HVAC breaker is tripped."
    ]
}


def get_next_troubleshooting_prompt(state: dict) -> str | None:
    """Returns the next troubleshooting prompt, or None if no more steps."""
    appliance = state.get("appliance_type")
    step_index = state.get("troubleshooting_step", 0)
    
    if appliance not in BASIC_TROUBLESHOOTING:
        return None
    
    steps = BASIC_TROUBLESHOOTING[appliance]
    if step_index >= len(steps):
        return None
    
    prompt = steps[step_index]
    state["troubleshooting_step"] = step_index + 1
    return prompt


def is_positive_response(user_text: str) -> bool:
    """Returns True if user response indicates success/yes."""
    text = user_text.lower()
    words = text.split()
    positive_words = {"yes", "yeah", "yep", "yup", "ok", "okay"}
    positive_phrases = ["it worked", "fixed", "working", "helped", "that helped"]
    
    if any(word in positive_words for word in words):
        return True
    if any(phrase in text for phrase in positive_phrases):
        return True
    return False
