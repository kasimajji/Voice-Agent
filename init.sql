-- MySQL Initialization Script for Voice AI Agent
-- This script creates all necessary tables and is executed automatically when the MySQL container starts

CREATE DATABASE IF NOT EXISTS voice_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE voice_ai;

-- Technicians table
CREATE TABLE IF NOT EXISTS technicians (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(50),
    email VARCHAR(255),
    INDEX idx_technician_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Technician service areas (ZIP code coverage)
CREATE TABLE IF NOT EXISTS technician_service_areas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    technician_id INT NOT NULL,
    zip_code VARCHAR(20) NOT NULL,
    FOREIGN KEY (technician_id) REFERENCES technicians(id) ON DELETE CASCADE,
    INDEX idx_zip_code (zip_code),
    INDEX idx_technician_id (technician_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Technician specialties (appliance types)
CREATE TABLE IF NOT EXISTS technician_specialties (
    id INT AUTO_INCREMENT PRIMARY KEY,
    technician_id INT NOT NULL,
    appliance_type VARCHAR(100) NOT NULL,
    FOREIGN KEY (technician_id) REFERENCES technicians(id) ON DELETE CASCADE,
    INDEX idx_appliance_type (appliance_type),
    INDEX idx_technician_id (technician_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Availability slots for scheduling
CREATE TABLE IF NOT EXISTS availability_slots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    technician_id INT NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME NOT NULL,
    is_booked BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (technician_id) REFERENCES technicians(id) ON DELETE CASCADE,
    INDEX idx_start_time (start_time),
    INDEX idx_is_booked (is_booked),
    INDEX idx_technician_id (technician_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Appointments
CREATE TABLE IF NOT EXISTS appointments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    call_sid VARCHAR(100),
    customer_phone VARCHAR(50),
    zip_code VARCHAR(20),
    appliance_type VARCHAR(100),
    symptom_summary VARCHAR(1000),
    error_codes VARCHAR(500),
    is_urgent BOOLEAN,
    technician_id INT,
    start_time DATETIME,
    end_time DATETIME,
    FOREIGN KEY (technician_id) REFERENCES technicians(id) ON DELETE SET NULL,
    INDEX idx_call_sid (call_sid),
    INDEX idx_customer_phone (customer_phone),
    INDEX idx_technician_id (technician_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Image upload tokens (Tier 3 vision analysis)
CREATE TABLE IF NOT EXISTS image_upload_tokens (
    id INT AUTO_INCREMENT PRIMARY KEY,
    token VARCHAR(64) UNIQUE NOT NULL,
    call_sid VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    appliance_type VARCHAR(100),
    symptom_summary TEXT,
    created_at DATETIME NOT NULL,
    expires_at DATETIME NOT NULL,
    used_at DATETIME,
    image_url VARCHAR(500),
    analysis_summary TEXT,
    troubleshooting_tips TEXT,
    is_appliance_image BOOLEAN DEFAULT NULL,
    INDEX idx_token (token),
    INDEX idx_call_sid (call_sid),
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Grant privileges (for Docker MySQL user)
GRANT ALL PRIVILEGES ON voice_ai.* TO 'voice_ai_user'@'%';
FLUSH PRIVILEGES;
