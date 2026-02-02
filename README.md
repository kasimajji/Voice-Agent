# Sears Home Services - Voice AI Agent

An intelligent voice-based customer support agent for home appliance troubleshooting, built with FastAPI, Twilio, and Google Gemini AI.

## 🎯 Features

- **Voice Interaction**: Natural phone-based conversations via Twilio
- **AI-Powered Troubleshooting**: 3-tier support system using Gemini 2.5 Flash
- **Image Analysis**: Upload appliance photos for visual diagnosis
- **Smart Scheduling**: Automated technician appointment booking
- **Email Notifications**: SendGrid integration for upload links

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Phone    │────▶│     Twilio      │────▶│   FastAPI App   │
│                 │◀────│   Voice API     │◀────│                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                │                                │
                        ▼                                ▼                                ▼
               ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
               │  Gemini 2.5     │             │    SQLite DB    │             │    SendGrid     │
               │  Flash LLM      │             │  (Scheduling)   │             │    (Email)      │
               └─────────────────┘             └─────────────────┘             └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Twilio Account (with phone number)
- Google AI API Key (Gemini)
- ngrok (for local development)

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/shs-voice-ai-agent.git
cd shs-voice-ai-agent

# Create environment file
cp app/.env.example app/.env
# Edit app/.env with your API keys
```

### 2. Environment Variables

Create `app/.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key
APP_BASE_URL=https://your-domain.ngrok-free.app
SENDGRID_API_KEY=your_sendgrid_key  # Optional
SENDGRID_FROM_EMAIL=noreply@yourdomain.com  # Optional
```

### 3. Launch with Docker

```bash
# Build and start
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### 4. Expose with ngrok (Development)

```bash
ngrok http 8000
# Copy the HTTPS URL to APP_BASE_URL in .env
```

### 5. Configure Twilio

1. Go to Twilio Console → Phone Numbers
2. Select your number → Voice Configuration
3. Set webhook URL: `https://your-domain.ngrok-free.app/twilio/voice`
4. Method: POST

### 6. Test

Call your Twilio phone number and describe an appliance issue!

## 📁 Project Structure

```
shs-voice-ai-agent/
├── app/
│   ├── main.py           # FastAPI application entry
│   ├── config.py         # Environment configuration
│   ├── db.py             # Database setup (SQLAlchemy)
│   ├── models.py         # Data models
│   ├── llm.py            # Gemini AI integration
│   ├── conversation.py   # Conversation state management
│   ├── twilio_routes.py  # Voice webhook handlers
│   ├── upload_routes.py  # Image upload endpoints
│   ├── image_service.py  # Image analysis service
│   └── seed.py           # Database seeding
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🎭 3-Tier Support Flow

### Tier 1: Basic Troubleshooting
- Common fixes (power cycle, check connections)
- No personal info required
- ~30 seconds

### Tier 2: Advanced Diagnosis
- Detailed symptom analysis
- Model-specific troubleshooting
- Brand and model detection

### Tier 3: Visual Analysis
- Email collection for photo upload
- Gemini Vision analyzes appliance images
- Specific repair recommendations

### Tier 4: Technician Scheduling
- ZIP code-based availability
- Real-time slot booking
- Confirmation with details

## � Service Coverage

**20 technicians** across **5 metro areas** covering **10 ZIP codes**:

| Metro Area | ZIP Codes |
|------------|-----------|
| Chicago | 60115, 60601, 60602, 60611 |
| New York | 10001, 10002, 11201 |
| San Francisco | 94105 |
| Dallas | 75201 |
| Atlanta | 30301 |

**Appliance Specialties**: Refrigerator, Washer, Dryer, Dishwasher, Oven, HVAC

## �🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/twilio/voice` | POST | Incoming call handler |
| `/twilio/voice/continue` | POST | Conversation continuation |
| `/upload/{token}` | GET | Upload page |
| `/upload/{token}` | POST | Image upload handler |
| `/upload/status/{call_sid}` | GET | Upload status check |

## 🧪 Local Development (Without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --port 8000
```

## 📊 Database

SQLite by default. For production, set `DATABASE_URL`:

```env
# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# SQLite (default)
DATABASE_URL=sqlite:///./voice_ai.db
```

## 🔒 Security Notes

- API keys are loaded from environment variables only
- `.env` files are excluded from Docker images
- Sensitive data never logged in production
- Upload tokens expire after 24 hours

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with ❤️ for Sears Home Services
