"""Tests for app.seed module — seed data integrity and advisory lock."""
import pytest
from app.seed import TECHNICIANS_DATA


class TestSeedData:
    def test_technician_count(self):
        assert len(TECHNICIANS_DATA) == 20

    def test_technician_data_structure(self):
        for entry in TECHNICIANS_DATA:
            name, phone, email, zip_codes, specialties = entry
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(phone, str)
            assert "@" in email
            assert isinstance(zip_codes, list) and len(zip_codes) > 0
            assert isinstance(specialties, list) and len(specialties) > 0

    def test_all_zip_codes_are_5_digits(self):
        for _, _, _, zip_codes, _ in TECHNICIANS_DATA:
            for z in zip_codes:
                assert len(z) == 5 and z.isdigit(), f"Invalid ZIP: {z}"

    def test_all_specialties_are_valid(self):
        valid = {"washer", "dryer", "refrigerator", "dishwasher", "oven", "hvac"}
        for _, _, _, _, specialties in TECHNICIANS_DATA:
            for s in specialties:
                assert s in valid, f"Invalid specialty: {s}"

    def test_unique_emails(self):
        emails = [email for _, _, email, _, _ in TECHNICIANS_DATA]
        assert len(emails) == len(set(emails)), "Duplicate emails in seed data"

    def test_unique_phones(self):
        phones = [phone for _, phone, _, _, _ in TECHNICIANS_DATA]
        assert len(phones) == len(set(phones)), "Duplicate phones in seed data"

    def test_geographic_coverage(self):
        """Verify all 10 ZIP codes are covered."""
        all_zips = set()
        for _, _, _, zip_codes, _ in TECHNICIANS_DATA:
            all_zips.update(zip_codes)
        expected = {"60115", "60601", "60602", "60611", "10001", "10002", "11201", "94105", "75201", "30301"}
        assert all_zips == expected
