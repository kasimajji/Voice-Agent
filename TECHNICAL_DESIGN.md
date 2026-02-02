# Technical Design Document
## Sears Home Services Voice AI Agent

---

## 1. System Overview

This document outlines the architectural decisions, tradeoffs, and rationale for the Voice AI Agent system designed to handle customer support calls for home appliance troubleshooting.

### Core Objectives
- Provide 24/7 automated troubleshooting support via phone
- Reduce call center load through AI-powered resolution
- Enable visual diagnosis through image uploads
- Seamlessly escalate to technician scheduling when needed

---

## 2. Architecture Decisions

### 2.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Backend Framework** | FastAPI | Async support, automatic OpenAPI docs, high performance, Python ecosystem |
| **Voice Platform** | Twilio | Industry leader, reliable STT/TTS, easy webhook integration, pay-per-use |
| **LLM** | Gemini 2.5 Flash | Multimodal (text + vision), fast inference, cost-effective, generous free tier |
| **Database** | SQLite → PostgreSQL | SQLite for dev simplicity, PostgreSQL for production scalability |
| **Email Service** | SendGrid | Reliable delivery, simple API, free tier for development |
| **Containerization** | Docker | Consistent environments, easy deployment, industry standard |

### 2.2 Why FastAPI over alternatives?

| Alternative | Why Not Chosen |
|-------------|----------------|
| Flask | No native async, slower for I/O-bound workloads |
| Django | Too heavyweight for a focused API service |
| Node.js/Express | Python better for AI/ML integration with Gemini |

### 2.3 Why Gemini over GPT-4 / Claude?

| Factor | Gemini 2.5 Flash | GPT-4 | Claude |
|--------|------------------|-------|--------|
| **Cost** | $0.075/1M tokens | $30/1M tokens | $15/1M tokens |
| **Vision** | ✅ Native | ✅ Native | ✅ Native |
| **Speed** | ~200ms | ~500ms | ~400ms |
| **Free Tier** | Generous | Limited | Limited |

**Decision**: Gemini provides the best cost-to-performance ratio for a voice application requiring fast responses.

---

## 3. Conversation Flow Design

### 3.1 State Machine Architecture

```
┌──────────────┐
│    START     │
└──────┬───────┘
       ▼
┌──────────────┐    Not Appliance    ┌──────────────┐
│  GREETING    │───────────────────▶│   REDIRECT   │
└──────┬───────┘                     └──────────────┘
       │ Appliance Detected
       ▼
┌──────────────┐
│   TIER 1     │──── Resolved ────▶ END
│  Basic Fix   │
└──────┬───────┘
       │ Not Resolved
       ▼
┌──────────────┐
│   TIER 2     │──── Resolved ────▶ END
│  Advanced    │
└──────┬───────┘
       │ Needs Visual
       ▼
┌──────────────┐
│   TIER 3     │──── Resolved ────▶ END
│ Image Upload │
└──────┬───────┘
       │ Needs Technician
       ▼
┌──────────────┐
│   TIER 4     │
│  Scheduling  │───────────────────▶ END
└──────────────┘
```

### 3.2 Why This Flow?

1. **Progressive Escalation**: Resolve simple issues quickly (60% of calls at Tier 1)
2. **Cost Optimization**: Avoid expensive technician visits when possible
3. **User Satisfaction**: Multiple resolution paths based on user preference
4. **Data Collection**: Gather symptoms before visual analysis for context

---

## 4. Key Design Tradeoffs

### 4.1 Twilio STT vs Google Speech-to-Text

| Approach | Pros | Cons |
|----------|------|------|
| **Twilio STT (Chosen)** | Zero additional cost, simple integration | Lower accuracy for numbers/spelling |
| Google Speech-to-Text | Higher accuracy | Additional API costs, complex WebSocket handling |

**Mitigation**: Implemented robust pre-processing (`_normalize_speech_for_email`) to handle STT errors deterministically before LLM processing.

### 4.2 Synchronous vs Asynchronous Image Analysis

| Approach | Pros | Cons |
|----------|------|------|
| **Polling (Chosen)** | Simple, works with Twilio's request model | Slightly longer wait time |
| WebSocket | Real-time updates | Complex, Twilio doesn't support natively |

**Decision**: Polling with 10-second intervals provides good UX while maintaining simplicity.

### 4.3 SQLite vs PostgreSQL

| Environment | Choice | Rationale |
|-------------|--------|-----------|
| Development | SQLite | Zero setup, single file, easy testing |
| Production | PostgreSQL | Concurrent access, better performance, Azure integration |

**Implementation**: `DATABASE_URL` environment variable allows seamless switching.

---

## 5. Scalability Considerations

### 5.1 Current Limitations
- Single SQLite database (dev mode)
- In-memory conversation state
- Local file storage for uploads

### 5.2 Production Scaling Path

```
Current                          Production
────────                         ──────────
SQLite           ──────▶         Azure PostgreSQL
Local files      ──────▶         Azure Blob Storage
In-memory state  ──────▶         Redis
Single instance  ──────▶         Azure Container Apps (auto-scale)
```

### 5.3 Estimated Capacity

| Configuration | Concurrent Calls | Monthly Cost |
|---------------|------------------|--------------|
| Single Container | 10-20 | ~$20 |
| 3 Containers + Redis | 50-100 | ~$100 |
| Auto-scaled | 500+ | ~$500+ |

---

## 6. Data Model

### 6.1 Seed Data Overview

| Entity | Count | Description |
|--------|-------|-------------|
| Technicians | 20 | Service professionals with contact info |
| Service Areas | 40 | Technician-to-ZIP mappings |
| Specialties | 32 | Technician-to-appliance mappings |
| Availability Slots | 400 | 10 days × 2 slots × 20 technicians |

### 6.2 Geographic Coverage

| Metro Area | ZIP Codes | Technicians |
|------------|-----------|-------------|
| Chicago | 60115, 60601, 60602, 60611 | 9 |
| New York | 10001, 10002, 11201 | 6 |
| San Francisco | 94105 | 3 |
| Dallas | 75201 | 4 |
| Atlanta | 30301 | 3 |

### 6.3 Appliance Coverage

All 6 major appliance categories covered:
- **Refrigerator**: 9 technicians
- **Washer**: 7 technicians
- **Dryer**: 6 technicians
- **Dishwasher**: 5 technicians
- **Oven**: 5 technicians
- **HVAC**: 6 technicians

---

## 7. Security Design

### 7.1 Secrets Management
- All API keys via environment variables
- `.env` files excluded from Docker images
- No secrets in source code or logs

### 7.2 Data Protection
- Upload tokens expire after 24 hours
- No PII stored beyond session
- HTTPS enforced for all endpoints

### 7.3 Input Validation
- Email regex validation + TLD whitelist
- ZIP code format validation
- Speech input sanitization before LLM

---

## 8. Error Handling Strategy

### 8.1 Graceful Degradation

| Failure | Fallback |
|---------|----------|
| Gemini API down | Generic troubleshooting tips |
| Image upload fails | Skip to scheduling |
| Email send fails | Log and continue flow |
| Database error | In-memory fallback for critical data |

### 8.2 Retry Logic
- Email confirmation: 3 attempts before fallback
- ZIP code collection: 3 attempts before agent transfer
- Image upload polling: 3 minutes timeout

---

## 9. Future Enhancements

### Phase 2 (Planned)
- [ ] Multi-language support (Spanish, French)
- [ ] SMS follow-up after calls
- [ ] Calendar integration for scheduling
- [ ] Call recording and analytics

### Phase 3 (Roadmap)
- [ ] Proactive outreach for maintenance reminders
- [ ] Integration with parts ordering system
- [ ] Voice biometrics for customer identification

---

## 10. Testing Strategy

| Level | Coverage | Tools |
|-------|----------|-------|
| Unit | LLM extraction, validation | pytest |
| Integration | Twilio webhooks | pytest + mocking |
| E2E | Full call flow | Manual + Twilio Test Credentials |

---

## 11. Conclusion

This architecture prioritizes:
1. **Simplicity**: Minimal moving parts for reliability
2. **Cost-efficiency**: Optimal model selection and progressive escalation
3. **User Experience**: Natural conversation flow with multiple resolution paths
4. **Maintainability**: Clean separation of concerns, documented code

The system is designed to handle the majority of common appliance issues automatically while providing a smooth path to human assistance when needed.

---

*Document Version: 1.0*  
*Last Updated: February 2026*
