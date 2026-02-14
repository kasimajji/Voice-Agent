from datetime import datetime
import json
from .db import SessionLocal
from .models import ConversationState


def _get_initial_state() -> dict:
    """Returns the initial state template."""
    return {
        "step": "greet_ask_name",
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
        # Email capture with confirmation loop (Issue 1)
        "customer_email": None,
        "pending_email": None,  # Email awaiting confirmation
        "email_attempts": 0,
        "email_confirm_attempts": 0,
        # Image upload flow (Issue 2)
        "image_upload_sent": False,
        "upload_token": None,
        "waiting_for_upload": False,
        "upload_poll_count": 0,
        "image_analysis_spoken": False,
        # Autonomous flow fields
        "understand_attempts": 0,
        "appliance_attempts": 0,
        "zip_attempts": 0,
        "customer_name": None,
        "troubleshooting_steps_text": None,
        "analysis_spoken": False,
        "upload_wait_attempts": 0,
    }


def _serialize_state(state: dict) -> dict:
    """Convert datetime objects in state to ISO format strings for JSON storage."""
    serialized = {}
    for key, value in state.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            # Handle lists - check if it's offered_slots (list of dicts with datetime)
            if key == "offered_slots":
                serialized[key] = []
                for slot in value:
                    if isinstance(slot, dict):
                        slot_copy = {}
                        for k, v in slot.items():
                            if isinstance(v, datetime):
                                slot_copy[k] = v.isoformat()
                            else:
                                slot_copy[k] = v
                        serialized[key].append(slot_copy)
                    else:
                        serialized[key].append(slot)
            else:
                # Regular list - just check for datetime objects
                serialized[key] = [
                    item.isoformat() if isinstance(item, datetime) else item
                    for item in value
                ]
        elif isinstance(value, dict):
            # Handle nested dicts
            serialized[key] = {
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in value.items()
            }
        else:
            serialized[key] = value
    return serialized


def _deserialize_state(state_data: dict) -> dict:
    """Convert ISO format strings back to datetime objects where needed."""
    deserialized = {}
    for key, value in state_data.items():
        if key in ["start_time", "end_time"] and isinstance(value, str):
            try:
                deserialized[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                deserialized[key] = value
        elif isinstance(value, list):
            # Handle lists - check if it's offered_slots
            if key == "offered_slots":
                deserialized[key] = []
                for slot in value:
                    if isinstance(slot, dict):
                        slot_copy = {}
                        for k, v in slot.items():
                            if k in ["start_time", "end_time"] and isinstance(v, str):
                                try:
                                    slot_copy[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                                except (ValueError, AttributeError):
                                    slot_copy[k] = v
                            else:
                                slot_copy[k] = v
                        deserialized[key].append(slot_copy)
                    else:
                        deserialized[key].append(slot)
            else:
                deserialized[key] = value  # Keep as-is for other lists
        elif isinstance(value, dict):
            # Handle nested dicts - convert datetime strings in offered_slots
            if key == "offered_slots":
                deserialized[key] = []
                for slot in value:
                    slot_copy = {}
                    for k, v in slot.items():
                        if k in ["start_time", "end_time"] and isinstance(v, str):
                            try:
                                slot_copy[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                            except (ValueError, AttributeError):
                                slot_copy[k] = v
                        else:
                            slot_copy[k] = v
                    deserialized[key].append(slot_copy)
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value
    return deserialized


def get_state(call_id: str) -> dict:
    """Returns existing state or initializes a new one for the call from database."""
    db = SessionLocal()
    try:
        # Try to get existing state from database
        state_record = db.query(ConversationState).filter(
            ConversationState.call_sid == call_id
        ).first()
        
        if state_record:
            # Return existing state (deserialize if needed)
            state_data = state_record.state_data
            if isinstance(state_data, dict):
                return _deserialize_state(state_data)
            return state_data
        
        # No existing state - create initial state
        initial_state = _get_initial_state()
        
        # Save to database (serialize datetime objects)
        serialized_state = _serialize_state(initial_state)
        new_state_record = ConversationState(
            call_sid=call_id,
            state_data=serialized_state,
            updated_at=datetime.utcnow()
        )
        db.add(new_state_record)
        db.commit()
        
        return initial_state
    except Exception as e:
        import logging
        logger = logging.getLogger("voice_agent.conversation")
        logger.error(f"Failed to get state for {call_id}: {e}", exc_info=True)
        db.rollback()
        # Fallback to initial state if DB fails
        return _get_initial_state()
    finally:
        db.close()


def update_state(call_id: str, new_state: dict) -> None:
    """Updates the state for a given call in database."""
    db = SessionLocal()
    try:
        # Serialize datetime objects before storing
        serialized_state = _serialize_state(new_state)
        
        state_record = db.query(ConversationState).filter(
            ConversationState.call_sid == call_id
        ).first()
        
        if state_record:
            # Update existing record
            state_record.state_data = serialized_state
            state_record.updated_at = datetime.utcnow()
        else:
            # Create new record if it doesn't exist
            state_record = ConversationState(
                call_sid=call_id,
                state_data=serialized_state,
                updated_at=datetime.utcnow()
            )
            db.add(state_record)
        
        db.commit()
    except Exception as e:
        import logging
        logger = logging.getLogger("voice_agent.conversation")
        logger.error(f"Failed to update state for {call_id}: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def infer_appliance_type(user_text: str) -> str | None:
    """Infers appliance type from user text using simple keyword matching."""
    text = user_text.lower()
    # Check "dishwasher" BEFORE "washer" to avoid false match
    if "dishwasher" in text:
        return "dishwasher"
    if "washer" in text or "washing machine" in text:
        return "washer"
    if "dryer" in text:
        return "dryer"
    if "fridge" in text or "refrigerator" in text:
        return "refrigerator"
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


def get_troubleshooting_steps_summary(appliance_type: str) -> str:
    """
    Return ALL troubleshooting steps for an appliance as a concise summary.
    Used in the autonomous flow where all steps are presented at once.
    
    Returns empty string if no steps are available for the appliance.
    """
    appliance = appliance_type.lower() if appliance_type else ""
    steps = BASIC_TROUBLESHOOTING.get(appliance, [])
    
    if not steps:
        return ""
    
    # Format as numbered list for speech
    parts = []
    for i, step in enumerate(steps, 1):
        # Remove "Please " prefix for conciseness when presenting all at once
        clean_step = step.replace("Please check", "Check").replace("Please ", "")
        parts.append(f"Step {i}: {clean_step}")
    
    return " ".join(parts)


def is_positive_response(user_text: str) -> bool:
    """Returns True if user response indicates success/yes.
    
    Handles negations like "not working", "didn't help", "no it's not fixed".
    """
    text = user_text.lower()
    words = text.split()
    
    # Check for explicit negative responses first
    negative_patterns = [
        "not working", "isn't working", "isn't helping", "not helping",
        "didn't work", "didn't help", "doesn't work", "doesn't help",
        "still not", "still broken", "still having", "no luck",
        "not fixed", "isn't fixed", "didn't fix", "doesn't fix",
        "same problem", "same issue", "no change", "nothing changed",
        "nope", "negative", "unfortunately"
    ]
    
    # If any negative pattern is found, return False immediately
    if any(pattern in text for pattern in negative_patterns):
        return False
    
    # Check for "no" at the start or standalone
    if text.strip().startswith("no") or text.strip() == "no":
        # But allow "no problem" or "no worries" as neutral
        if "no problem" not in text and "no worries" not in text:
            return False
    
    # Now check for positive responses
    positive_words = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "absolutely"}
    positive_phrases = ["it worked", "that worked", "fixed", "working now", 
                        "helped", "that helped", "it's working", "all good",
                        "problem solved", "issue resolved", "good now"]
    
    if any(word in positive_words for word in words):
        return True
    if any(phrase in text for phrase in positive_phrases):
        return True
    return False
