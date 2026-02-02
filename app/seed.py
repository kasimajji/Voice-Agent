from datetime import datetime, timedelta
from .db import SessionLocal
from .models import Technician, TechnicianServiceArea, TechnicianSpecialty, AvailabilitySlot


# Technician data: name, phone, email, zip_codes, specialties
TECHNICIANS_DATA = [
    ("Alex Martinez", "555-1001", "alex.martinez@searshomeservices.com", 
     ["60601", "60602", "60603"], ["refrigerator", "washer"]),
    ("Maria Chen", "555-1002", "maria.chen@searshomeservices.com", 
     ["10001", "10002", "60601"], ["washer", "dryer"]),
    ("John Patel", "555-1003", "john.patel@searshomeservices.com", 
     ["60601", "10001", "90210"], ["dryer", "dishwasher", "oven"]),
    ("Priya Singh", "555-1004", "priya.singh@searshomeservices.com", 
     ["90210", "90211", "90212"], ["refrigerator", "hvac"]),
    ("David Johnson", "555-1005", "david.johnson@searshomeservices.com", 
     ["60601", "60602", "10001"], ["hvac", "oven"]),
    ("Emily Clark", "555-1006", "emily.clark@searshomeservices.com", 
     ["10001", "10002", "10003"], ["washer", "dryer", "dishwasher"]),
    ("Michael Brown", "555-1007", "michael.brown@searshomeservices.com", 
     ["90210", "60601", "77001"], ["refrigerator", "oven"]),
    ("Sarah Lopez", "555-1008", "sarah.lopez@searshomeservices.com", 
     ["77001", "77002", "77003"], ["washer", "dryer", "hvac"]),
    ("Kevin Nguyen", "555-1009", "kevin.nguyen@searshomeservices.com", 
     ["60601", "60602", "77001"], ["dishwasher", "oven", "refrigerator"]),
    ("Laura Garcia", "555-1010", "laura.garcia@searshomeservices.com", 
     ["10001", "90210", "77001"], ["hvac", "washer", "dryer"]),
]

# ZIP codes covered: 60601, 60602, 60603, 10001, 10002, 10003, 90210, 90211, 90212, 77001, 77002, 77003


def seed_data():
    """Seed database with 10 technicians, service areas, specialties, and availability slots."""
    db = SessionLocal()
    try:
        if db.query(Technician).first():
            print("[Seed] Data already exists, skipping seed.")
            return

        technicians = []
        for name, phone, email, _, _ in TECHNICIANS_DATA:
            tech = Technician(name=name, phone=phone, email=email)
            technicians.append(tech)
        
        db.add_all(technicians)
        db.commit()
        
        # Refresh to get IDs
        for tech in technicians:
            db.refresh(tech)

        # Add service areas and specialties
        service_areas = []
        specialties = []
        
        for i, (_, _, _, zip_codes, appliance_types) in enumerate(TECHNICIANS_DATA):
            tech_id = technicians[i].id
            for zip_code in zip_codes:
                service_areas.append(TechnicianServiceArea(technician_id=tech_id, zip_code=zip_code))
            for appliance_type in appliance_types:
                specialties.append(TechnicianSpecialty(technician_id=tech_id, appliance_type=appliance_type))

        db.add_all(service_areas + specialties)

        # Create availability slots for next 10 days (morning + afternoon per tech)
        now = datetime.utcnow()
        slots = []
        for tech in technicians:
            for day_offset in range(1, 11):  # Next 10 days
                # Morning slot: 9 AM - 12 PM
                morning_start = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
                morning_end = morning_start + timedelta(hours=3)
                slots.append(AvailabilitySlot(
                    technician_id=tech.id,
                    start_time=morning_start,
                    end_time=morning_end,
                    is_booked=False
                ))
                # Afternoon slot: 1 PM - 4 PM
                afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0) + timedelta(days=day_offset)
                afternoon_end = afternoon_start + timedelta(hours=3)
                slots.append(AvailabilitySlot(
                    technician_id=tech.id,
                    start_time=afternoon_start,
                    end_time=afternoon_end,
                    is_booked=False
                ))

        db.add_all(slots)
        db.commit()
        
        print(f"[Seed] Database seeded: {len(technicians)} technicians, "
              f"{len(service_areas)} service areas, {len(specialties)} specialties, "
              f"{len(slots)} availability slots.")
    except Exception as e:
        print(f"[Seed Error] Failed to seed database: {e}")
        db.rollback()
    finally:
        db.close()
