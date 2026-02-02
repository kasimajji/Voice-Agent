from sqlalchemy import func, select
from .db import SessionLocal
from .models import AvailabilitySlot, Appointment, TechnicianServiceArea, TechnicianSpecialty, Technician
from datetime import datetime


def find_available_slots(zip_code: str, appliance_type: str, time_preference: str = None, limit: int = 3):
    """
    Find available technician slots based on ZIP code, appliance type, and optional time preference.
    
    Args:
        zip_code: Customer's ZIP code
        appliance_type: Type of appliance needing service
        time_preference: 'morning' or 'afternoon' (optional)
        limit: Maximum number of slots to return
    
    Returns:
        List of available slot dictionaries
    """
    db = SessionLocal()
    try:
        # Query available slots matching ZIP code and appliance specialty
        query = (
            db.query(AvailabilitySlot)
            .join(Technician)
            .filter(
                AvailabilitySlot.technician_id.in_(
                    select(TechnicianServiceArea.technician_id).where(
                        TechnicianServiceArea.zip_code == zip_code
                    )
                ),
                AvailabilitySlot.technician_id.in_(
                    select(TechnicianSpecialty.technician_id).where(
                        TechnicianSpecialty.appliance_type == appliance_type
                    )
                ),
                AvailabilitySlot.is_booked == False,
                AvailabilitySlot.start_time > datetime.utcnow()
            )
        )
        
        # Filter by time preference if specified
        if time_preference == "morning":
            query = query.filter(
                func.extract('hour', AvailabilitySlot.start_time) < 12
            )
        elif time_preference == "afternoon":
            query = query.filter(
                func.extract('hour', AvailabilitySlot.start_time) >= 12
            )
        
        query = query.order_by(AvailabilitySlot.start_time).limit(limit)
        
        results = []
        for slot in query.all():
            results.append({
                "slot_id": slot.id,
                "technician_name": slot.technician.name,
                "technician_id": slot.technician.id,
                "start_time": slot.start_time,
                "end_time": slot.end_time
            })
        return results
    finally:
        db.close()


def book_appointment(call_sid: str, customer_phone: str, zip_code: str, appliance_type: str,
                     symptom_summary: str, error_codes: list, is_urgent: bool, chosen_slot_id: int):
    """
    Book an appointment by marking the slot as booked and creating an appointment record.
    
    Args:
        call_sid: Twilio call SID
        customer_phone: Customer's phone number
        zip_code: Customer's ZIP code
        appliance_type: Type of appliance
        symptom_summary: Summary of the issue
        error_codes: List of error codes
        is_urgent: Whether the issue is urgent
        chosen_slot_id: ID of the chosen availability slot
    
    Returns:
        The created Appointment object
    """
    db = SessionLocal()
    try:
        slot = db.query(AvailabilitySlot).filter_by(id=chosen_slot_id).first()
        if not slot:
            raise ValueError(f"Slot {chosen_slot_id} not found")
        if slot.is_booked:
            raise ValueError(f"Slot {chosen_slot_id} is already booked")
        
        slot.is_booked = True

        appt = Appointment(
            call_sid=call_sid,
            customer_phone=customer_phone,
            zip_code=zip_code,
            appliance_type=appliance_type,
            symptom_summary=symptom_summary,
            error_codes=",".join(error_codes) if error_codes else "",
            is_urgent=is_urgent,
            technician_id=slot.technician_id,
            start_time=slot.start_time,
            end_time=slot.end_time
        )

        db.add(appt)
        db.commit()
        db.refresh(appt)
        
        # Get technician name for return
        tech_name = slot.technician.name
        appt_info = {
            "id": appt.id,
            "technician_name": tech_name,
            "start_time": appt.start_time,
            "end_time": appt.end_time
        }
        return appt_info
    finally:
        db.close()


def format_slot_for_speech(slot: dict, option_number: int) -> str:
    """Format a slot dictionary for text-to-speech output."""
    start = slot["start_time"]
    day_name = start.strftime("%A")
    date_str = start.strftime("%B %d")
    
    hour = start.hour
    if hour < 12:
        time_of_day = "morning"
        time_str = f"{hour} AM" if hour > 0 else "12 AM"
    else:
        time_of_day = "afternoon"
        hour_12 = hour - 12 if hour > 12 else hour
        time_str = f"{hour_12} PM" if hour_12 > 0 else "12 PM"
    
    return f"Option {option_number}: {day_name}, {date_str}, {time_of_day} at {time_str} with {slot['technician_name']}"
