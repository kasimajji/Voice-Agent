from datetime import datetime, timedelta
from .db import SessionLocal
from .models import Technician, TechnicianServiceArea, TechnicianSpecialty, AvailabilitySlot


def seed_data():
    db = SessionLocal()
    try:
        if db.query(Technician).first():
            print("[Seed] Data already exists, skipping seed.")
            return

        t1 = Technician(name="Alex Martinez", phone="123", email="alex@example.com")
        t2 = Technician(name="Maria Chen", phone="456", email="maria@example.com")
        t3 = Technician(name="John Patel", phone="789", email="john@example.com")
        db.add_all([t1, t2, t3])
        db.commit()

        service_areas = [
            TechnicianServiceArea(technician_id=t1.id, zip_code="60601"),
            TechnicianServiceArea(technician_id=t1.id, zip_code="60602"),
            TechnicianServiceArea(technician_id=t2.id, zip_code="10001"),
            TechnicianServiceArea(technician_id=t2.id, zip_code="60601"),
            TechnicianServiceArea(technician_id=t3.id, zip_code="60601"),
            TechnicianServiceArea(technician_id=t3.id, zip_code="10001"),
        ]

        specialties = [
            TechnicianSpecialty(technician_id=t1.id, appliance_type="refrigerator"),
            TechnicianSpecialty(technician_id=t1.id, appliance_type="washer"),
            TechnicianSpecialty(technician_id=t2.id, appliance_type="washer"),
            TechnicianSpecialty(technician_id=t2.id, appliance_type="dryer"),
            TechnicianSpecialty(technician_id=t3.id, appliance_type="dryer"),
            TechnicianSpecialty(technician_id=t3.id, appliance_type="dishwasher"),
            TechnicianSpecialty(technician_id=t3.id, appliance_type="oven"),
            TechnicianSpecialty(technician_id=t3.id, appliance_type="hvac"),
        ]

        db.add_all(service_areas + specialties)

        now = datetime.utcnow()
        slots = []
        for tech in [t1, t2, t3]:
            for i in range(5):
                # Morning slot
                morning_start = now + timedelta(days=i+1, hours=9)
                morning_end = morning_start + timedelta(hours=3)
                slots.append(AvailabilitySlot(
                    technician_id=tech.id,
                    start_time=morning_start,
                    end_time=morning_end,
                    is_booked=False
                ))
                # Afternoon slot
                afternoon_start = now + timedelta(days=i+1, hours=13)
                afternoon_end = afternoon_start + timedelta(hours=3)
                slots.append(AvailabilitySlot(
                    technician_id=tech.id,
                    start_time=afternoon_start,
                    end_time=afternoon_end,
                    is_booked=False
                ))

        db.add_all(slots)
        db.commit()
        print("[Seed] Database seeded with technicians, service areas, specialties, and availability slots.")
    finally:
        db.close()
