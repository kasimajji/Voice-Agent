from datetime import datetime, timedelta
import logging
from .db import SessionLocal
from .models import Technician, TechnicianServiceArea, TechnicianSpecialty, AvailabilitySlot

logger = logging.getLogger("voice_agent.seed")


# Technician data: name, phone, email, zip_codes, specialties
# 20 technicians covering 10 ZIP codes across 5 metro areas
TECHNICIANS_DATA = [
    # Chicago Metro (60115, 60601, 60602, 60611)
    ("Alex Martinez", "555-1001", "alex.martinez@example.com", 
     ["60601", "60115"], ["refrigerator", "washer"]),
    ("Maria Chen", "555-1002", "maria.chen@example.com", 
     ["60601", "60602"], ["washer", "dryer"]),
    ("John Patel", "555-1003", "john.patel@example.com", 
     ["60115", "60611"], ["refrigerator"]),
    ("Kevin Nguyen", "555-1009", "kevin.nguyen@example.com", 
     ["60115", "60602"], ["dishwasher", "refrigerator"]),
    ("Laura Garcia", "555-1010", "laura.garcia@example.com", 
     ["60601", "60611"], ["dryer", "washer"]),
    ("Anika Sharma", "555-1012", "anika.sharma@example.com", 
     ["60115", "60602"], ["refrigerator"]),
    ("Omar Hassan", "555-1015", "omar.hassan@example.com", 
     ["60601", "60611"], ["hvac", "refrigerator"]),
    ("Chloe Taylor", "555-1016", "chloe.taylor@example.com", 
     ["30301", "60115"], ["dishwasher"]),
    ("Sophia Lee", "555-1020", "sophia.lee@example.com", 
     ["10002", "60115"], ["dishwasher", "refrigerator"]),
    
    # New York Metro (10001, 10002, 11201) + cross-region coverage
    ("Priya Singh", "555-1004", "priya.singh@example.com", 
     ["10001", "10002"], ["dishwasher", "oven"]),
    ("David Johnson", "555-1005", "david.johnson@example.com", 
     ["10001", "11201"], ["hvac", "refrigerator"]),
    ("Robert Wilson", "555-1011", "robert.wilson@example.com", 
     ["10001", "11201"], ["hvac"]),
    ("Jessica Miller", "555-1014", "jessica.miller@example.com", 
     ["10002", "94105"], ["dryer"]),  # NY + SF coverage
    ("Nina Rossi", "555-1018", "nina.rossi@example.com", 
     ["10001", "60602"], ["dryer", "oven"]),  # NY + Chicago coverage
    ("Ethan Walker", "555-1017", "ethan.walker@example.com", 
     ["11201", "75201"], ["refrigerator", "washer"]),  # NY + Dallas coverage
    
    # San Francisco (94105)
    ("Emily Clark", "555-1006", "emily.clark@example.com", 
     ["60601", "94105"], ["washer"]),
    ("Michael Brown", "555-1007", "michael.brown@example.com", 
     ["75201", "94105"], ["dryer", "hvac"]),
    ("Victor Kim", "555-1019", "victor.kim@example.com", 
     ["60601", "94105"], ["hvac"]),
    
    # Dallas (75201)
    ("Sarah Lopez", "555-1008", "sarah.lopez@example.com", 
     ["30301", "75201"], ["oven"]),
    ("Daniel Evans", "555-1013", "daniel.evans@example.com", 
     ["30301", "75201"], ["oven", "washer"]),
]

# ZIP codes covered (10 total across 5 metro areas):
# - Chicago: 60115, 60601, 60602, 60611
# - New York: 10001, 10002, 11201
# - San Francisco: 94105
# - Dallas: 75201
# - Atlanta: 30301


def seed_data():
    """Seed database with 20 technicians, service areas, specialties, and availability slots."""
    db = SessionLocal()
    try:
        if db.query(Technician).first():
            logger.info("Data already exists, skipping seed.")
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
        
        logger.info(f"Database seeded: {len(technicians)} technicians, "
              f"{len(service_areas)} service areas, {len(specialties)} specialties, "
              f"{len(slots)} availability slots.")
    except Exception as e:
        logger.error(f"Failed to seed database: {e}")
        db.rollback()
    finally:
        db.close()
