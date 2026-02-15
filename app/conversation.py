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


