# 🏠 Voice AI Home Appliance Support Assistant

An end-to-end voice agent that handles inbound calls, diagnoses appliance issues via conversation, and schedules technicians. Built for the **Sears Home Services AI Engineer** take-home assessment.

---

## 📞 Live Demo

| | |
|---|---|
| **Phone Number** | +1-XXX-XXX-XXXX |
| **Tech Stack** | FastAPI · Twilio · Gemini (text + vision) · MySQL · SendGrid · Docker |

> Call the number above to experience the full voice agent flow!

---

## ✨ Features

### Tier 1 – Core Voice Agent
- **Inbound call handling** – Twilio webhook integration for real-time voice interactions
- **Appliance identification** – LLM-powered classification of appliance types from natural speech
- **Multi-turn symptom collection** – Contextual follow-up questions to gather issue details
- **Troubleshooting flows with conversation memory** – State machine maintains context across turns

### Tier 2 – Technician Scheduling
- **Technician database** – 20 technicians across 5 metro areas with real availability slots
- **ZIP + appliance matching** – Finds technicians by location and specialty
- **Appointment slot offering and booking** – Voice-guided slot selection with confirmation

### Tier 3 – Visual Diagnosis
- **Email capture and secure upload link** – Robust speech-to-text email extraction with confirmation loop
- **Image ingestion endpoint** – Secure token-based upload page
- **Gemini Vision-based analysis** – AI analyzes appliance photos and provides specific troubleshooting advice

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HIGH-LEVEL COMPONENTS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   📞 Twilio          ──►   🚀 FastAPI Webhooks                              │
│   (Inbound Calls)          (Voice Handler)                                  │
│                                   │                                         │
│                                   ▼                                         │
│                         🤖 LLM Orchestrator (Gemini)                        │
│                         - Name extraction                                   │
│                         - Appliance classification                          │
│                         - Symptom analysis                                  │
│                         - Vision analysis                                   │
│                                   │                                         │
│                                   ▼                                         │
│                      💬 Conversation State Manager                          │
│                      (In-memory state machine)                              │
│                                   │                                         │
│                    ┌──────────────┼──────────────┐                          │
│                    ▼              ▼              ▼                          │
│              🗄️ MySQL DB    📧 SendGrid    🖼️ Image Upload                  │
│              (Scheduling)   (Email Links)  (Vision Analysis)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Python 3.11+** | Runtime environment |
| **Docker & docker-compose** | Container orchestration |
| **Twilio Account** | Phone number + API credentials |
| **Google AI API Key** | Gemini 2.5 Flash for LLM + Vision |
| **SendGrid API Key** | (Optional) For email delivery |
| **ngrok Account** | (Free) For local tunnel |

### One-Command Setup

```bash
# 1. Clone the repository
git clone https://github.com/kasimajji/Voice-Agent.git
cd Voice-Agent

# 2. Create environment file
cp app/.env.example app/.env

# 3. Edit app/.env with your credentials (see below)

# 4. Launch everything with ONE command
docker-compose up --build
```

**That's it!** The system will automatically:
- ✅ Start MySQL 8.0 database
- ✅ Initialize schema and seed technician data
- ✅ Start the FastAPI voice agent
- ✅ Create ngrok tunnel for public URL
- ✅ Update Twilio webhook automatically

---

## 🔐 Environment Configuration

Create `app/.env` with the following variables:

```env
# ============================================================================
# REQUIRED CREDENTIALS
# ============================================================================

# ngrok Authentication (get free token at https://ngrok.com)
NGROK_AUTHTOKEN=your_ngrok_auth_token

# Twilio Credentials (from Twilio Console)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER_SID=PNxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini API Key (from Google AI Studio)
GOOGLE_API_KEY=your_gemini_api_key

# ============================================================================
# OPTIONAL CREDENTIALS
# ============================================================================

# SendGrid for email delivery (optional - falls back to console logging)
SENDGRID_API_KEY=SG.your_sendgrid_api_key
SENDGRID_FROM_EMAIL=noreply@yourdomain.com

# App Base URL (auto-configured by ngrok, but can override)
APP_BASE_URL=http://localhost:8000

# ============================================================================
# DATABASE (Auto-configured for Docker - no changes needed)
# ============================================================================
DB_HOST=localhost
DB_PORT=3306
DB_USER=voice_ai_user
DB_PASSWORD=voiceaipassword
DB_NAME=voice_ai
DB_ROOT_PASSWORD=rootpassword
```

### Where to Get Credentials

| Credential | Where to Find |
|------------|---------------|
| `NGROK_AUTHTOKEN` | [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) → Your Authtoken |
| `TWILIO_ACCOUNT_SID` | [Twilio Console](https://console.twilio.com/) → Account Info |
| `TWILIO_AUTH_TOKEN` | [Twilio Console](https://console.twilio.com/) → Account Info |
| `TWILIO_PHONE_NUMBER_SID` | [Twilio Console](https://console.twilio.com/us1/develop/phone-numbers/manage/incoming) → Click number → Phone Number SID |
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) → Create API Key |
| `SENDGRID_API_KEY` | [SendGrid](https://app.sendgrid.com/settings/api_keys) → Create API Key |

---

## 📁 Project Structure

```
Voice-Agent/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Environment configuration
│   ├── db.py                # SQLAlchemy database setup
│   ├── models.py            # Database models (Technician, Appointment, etc.)
│   ├── llm.py               # Gemini LLM integration (text + vision)
│   ├── conversation.py      # Conversation state machine
│   ├── twilio_routes.py     # Voice webhook handlers (main call flow)
│   ├── upload_routes.py     # Image upload endpoints
│   ├── image_service.py     # Image analysis & email service
│   ├── scheduling.py        # Technician availability & booking
│   ├── logging_config.py    # Structured logging
│   ├── seed.py              # Database seeding (20 technicians)
│   └── .env.example         # Environment template
├── templates/
│   └── upload.html          # Image upload page
├── uploads/                 # Uploaded images directory
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile               # Application container
├── init.sql                 # Database schema
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🎭 Support Tiers

### Tier 1: Conversational Troubleshooting

| Feature | Description |
|---------|-------------|
| **Natural Greeting** | Personalized welcome with name capture |
| **Appliance Classification** | LLM identifies washer, dryer, fridge, dishwasher, oven, HVAC |
| **Symptom Extraction** | Structured analysis of error codes, noises, behaviors |
| **Guided Troubleshooting** | Step-by-step fixes with yes/no confirmation |

**Example Flow:**
```
Agent: "Hi! Thanks for calling Sears Home Services. May I have your name?"
User:  "Hi, my name is John"
Agent: "Nice to meet you, John! How are you doing today?"
User:  "Good, my refrigerator is making a loud noise"
Agent: "Got it, John! So you're having trouble with your refrigerator..."
```

### Tier 2: Technician Scheduling

| Feature | Description |
|---------|-------------|
| **ZIP Code Collection** | Voice-based location capture |
| **Availability Lookup** | Real-time slot search by ZIP + appliance |
| **Slot Selection** | "Option 1, 2, or 3" voice selection |
| **Booking Confirmation** | Appointment details read back |

**Coverage Areas:**

| Metro Area | ZIP Codes |
|------------|-----------|
| Chicago | 60115, 60601, 60602, 60611 |
| New York | 10001, 10002, 11201 |
| San Francisco | 94105 |
| Dallas | 75201 |
| Atlanta | 30301 |

### Tier 3: Visual Diagnosis

| Feature | Description |
|---------|-------------|
| **Email Capture** | Robust STT with spelling confirmation |
| **Secure Upload Link** | Token-based URL sent via email |
| **Image Validation** | Checks if image shows an appliance |
| **Vision Analysis** | Gemini analyzes photo for issues |
| **Specific Recommendations** | Tailored troubleshooting based on visual inspection |

**Example Flow:**
```
Agent: "I can send you a link to upload a photo. What's your email?"
User:  "j o h n at gmail dot com"
Agent: "I heard J-O-H-N at G-M-A-I-L dot C-O-M. Is that correct?"
User:  "Yes"
Agent: "I've sent an upload link to your email..."
[User uploads photo]
Agent: "I've analyzed your image. I can see frost buildup on the evaporator coils..."
```

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/twilio/voice` | POST | Incoming call webhook |
| `/twilio/voice/continue` | POST | Conversation continuation |
| `/upload/{token}` | GET | Upload page (HTML) |
| `/upload/{token}` | POST | Image upload handler |
| `/upload/status/{call_sid}` | GET | Upload/analysis status |

---

## 🧪 Testing the System

### Demo Script

1. **Call the Twilio number** and wait for greeting
2. **Say your name** when prompted
3. **Describe an issue:** "My refrigerator is making a loud humming noise"
4. **Follow troubleshooting steps** - say "yes" or "no" to each
5. **When offered image upload**, say "upload a photo"
6. **Spell your email:** "j o h n at gmail dot com"
7. **Confirm the email** when read back
8. **Check your email** and upload an appliance photo
9. **Listen to the AI analysis** of your image
10. **Say "schedule a technician"** to test booking
11. **Provide ZIP code:** 60601 (Chicago coverage)
12. **Select a time slot:** "Option 1"

### Test ZIP Codes

```
60601, 60602, 60611, 60115  (Chicago)
10001, 10002, 11201         (New York)
94105                       (San Francisco)
75201                       (Dallas)
30301                       (Atlanta)
```

---

## 🐳 Docker Commands

```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f voice-ai
docker-compose logs -f twilio-config

# Stop all services
docker-compose down

# Reset database (delete volume)
docker-compose down -v
docker-compose up --build
```

---

## 🖥️ Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MySQL locally and update .env with credentials

# Run the application
uvicorn app.main:app --reload --port 8000

# In another terminal, start ngrok
ngrok http 8000

# Update APP_BASE_URL in .env with ngrok URL
# Configure Twilio webhook manually
```

---

## 🔒 Security

- ✅ API keys loaded from environment variables only
- ✅ `.env` files excluded from Git and Docker images
- ✅ Upload tokens expire after 24 hours
- ✅ No sensitive data logged in production
- ✅ Secure HTTPS via ngrok tunnel

---

## 📊 Monitoring

| Dashboard | URL |
|-----------|-----|
| **ngrok Inspector** | http://localhost:4040 |
| **Health Check** | http://localhost:8000/health |
| **Docker Logs** | `docker-compose logs -f` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Kasim Ajji**

Built for the Sears Home Services AI Engineer Assessment
