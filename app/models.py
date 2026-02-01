from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base


class Technician(Base):
    __tablename__ = "technicians"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    phone = Column(String(50))
    email = Column(String(255))

    service_areas = relationship("TechnicianServiceArea", back_populates="technician")
    specialties = relationship("TechnicianSpecialty", back_populates="technician")
    slots = relationship("AvailabilitySlot", back_populates="technician")


class TechnicianServiceArea(Base):
    __tablename__ = "technician_service_areas"
    id = Column(Integer, primary_key=True)
    technician_id = Column(Integer, ForeignKey("technicians.id"))
    zip_code = Column(String(20), nullable=False)

    technician = relationship("Technician", back_populates="service_areas")


class TechnicianSpecialty(Base):
    __tablename__ = "technician_specialties"
    id = Column(Integer, primary_key=True)
    technician_id = Column(Integer, ForeignKey("technicians.id"))
    appliance_type = Column(String(100), nullable=False)

    technician = relationship("Technician", back_populates="specialties")


class AvailabilitySlot(Base):
    __tablename__ = "availability_slots"
    id = Column(Integer, primary_key=True)
    technician_id = Column(Integer, ForeignKey("technicians.id"))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    is_booked = Column(Boolean, default=False)

    technician = relationship("Technician", back_populates="slots")


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True)
    call_sid = Column(String(100))
    customer_phone = Column(String(50))
    zip_code = Column(String(20))
    appliance_type = Column(String(100))
    symptom_summary = Column(String(1000))
    error_codes = Column(String(500))
    is_urgent = Column(Boolean)
    technician_id = Column(Integer, ForeignKey("technicians.id"))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
