"""
Google STT v2 streaming transcription via Twilio Media Streams.

Architecture:
1. Twilio sends real-time audio via WebSocket (base64 μ-law 8kHz)
2. This module decodes and forwards audio chunks to Google STT v2 streaming API
3. Final transcripts are stored per-call and consumed by the state machine
4. Each utterance triggers a new STT streaming session (turn-based)

Usage:
- FastAPI WebSocket endpoint at /twilio/media-stream
- Twilio <Start><Stream> in TwiML points to this endpoint
- State machine reads transcripts via get_transcript() / clear_transcript()
"""

import asyncio
import base64
import json
import queue
import threading
import time
from collections import defaultdict
from typing import Optional

from google.cloud import speech
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import (
    STT_CONFIDENCE_THRESHOLD,
    STT_SILENCE_TIMEOUT_MS,
    STT_LANGUAGE_CODE,
    USE_STREAMING_STT,
)
from .logging_config import get_logger

logger = get_logger("stt_stream")

# ── Per-call transcript storage ──────────────────────────────────────────────
# Key: call_sid → dict with transcript, confidence, timestamp
_transcripts: dict[str, dict] = {}
_transcript_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)


def get_transcript(call_sid: str, not_before: float = 0.0) -> Optional[dict]:
    """
    Get the latest transcript for a call (non-blocking).
    If not_before is set, only return transcripts that arrived AFTER that timestamp.
    This prevents stale transcripts from a previous turn overriding the current one.
    """
    t = _transcripts.get(call_sid)
    if t and not_before > 0 and t.get("timestamp", 0) < not_before:
        logger.debug(
            f"Ignoring stale Google STT transcript (ts={t['timestamp']:.1f} < turn_start={not_before:.1f})",
            extra={"call_sid": call_sid},
        )
        return None
    return t


async def wait_for_transcript(call_sid: str, timeout: float = 30.0) -> Optional[dict]:
    """Wait for a transcript to arrive for a call (async, with timeout)."""
    event = _transcript_events[call_sid]
    event.clear()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return _transcripts.get(call_sid)
    except asyncio.TimeoutError:
        logger.warning(f"Transcript wait timed out for {call_sid[:16]}...")
        return None


def clear_transcript(call_sid: str):
    """Clear transcript after it's been consumed by the state machine."""
    _transcripts.pop(call_sid, None)
    if call_sid in _transcript_events:
        _transcript_events[call_sid].clear()


def _store_transcript(call_sid: str, text: str, confidence: float):
    """Store a final transcript and signal waiters."""
    _transcripts[call_sid] = {
        "text": text,
        "confidence": confidence,
        "timestamp": time.time(),
    }
    _transcript_events[call_sid].set()
    if text:
        logger.info(
            f"Google STT transcript: '{text[:80]}' (confidence={confidence:.2f})",
            extra={"call_sid": call_sid},
        )


# ── Google STT v2 streaming recognizer ───────────────────────────────────────

def _build_recognition_config() -> speech.RecognitionConfig:
    """Build Google STT recognition config for telephony audio."""
    return speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        language_code=STT_LANGUAGE_CODE,
        model="telephony",
        use_enhanced=True,
        enable_automatic_punctuation=True,
        speech_contexts=[
            speech.SpeechContext(
                phrases=[
                    "refrigerator", "washer", "dryer", "dishwasher", "oven",
                    "HVAC", "air conditioner", "furnace", "heat pump",
                    "not cooling", "not heating", "leaking", "making noise",
                    "error code", "schedule", "technician", "appointment",
                    "morning", "afternoon", "evening",
                    "yes", "no", "correct", "wrong",
                    "gmail", "yahoo", "outlook", "hotmail", "at", "dot com",
                ],
                boost=15.0,
            )
        ],
    )


def _build_streaming_config() -> speech.StreamingRecognitionConfig:
    """Build streaming config wrapping the recognition config."""
    return speech.StreamingRecognitionConfig(
        config=_build_recognition_config(),
        interim_results=False,
        single_utterance=False,
    )


def _run_stt_on_audio(audio_data: bytes, call_sid: str) -> tuple[str, float]:
    """
    Run synchronous streaming recognition on collected audio bytes.
    Uses the correct v2 SDK signature: streaming_recognize(config, requests).
    Returns (transcript_text, confidence).
    """
    client = speech.SpeechClient()
    streaming_config = _build_streaming_config()

    def audio_generator():
        """Yield audio in ~3200-byte chunks (0.2s at 8kHz μ-law)."""
        chunk_size = 3200
        for i in range(0, len(audio_data), chunk_size):
            yield speech.StreamingRecognizeRequest(
                audio_content=audio_data[i : i + chunk_size]
            )

    best_text = ""
    best_confidence = 0.0

    try:
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=audio_generator(),
        )
        for resp in responses:
            for result in resp.results:
                if result.is_final and result.alternatives:
                    alt = result.alternatives[0]
                    if alt.confidence >= best_confidence:
                        best_text = alt.transcript.strip()
                        best_confidence = alt.confidence
    except Exception as e:
        logger.error(f"Google STT recognize error: {e}", extra={"call_sid": call_sid})

    return best_text, best_confidence


async def _process_audio_continuously(call_sid: str, audio_queue: asyncio.Queue):
    """
    Continuously collect audio chunks and periodically run STT.
    Stores transcripts as they arrive so the state machine can consume them.

    Strategy: Collect audio in windows. When silence is detected (no new chunks
    for STT_SILENCE_TIMEOUT_MS), run STT on the accumulated audio, store the
    transcript, and reset the buffer for the next utterance.
    """
    audio_buffer = bytearray()
    silence_timeout = STT_SILENCE_TIMEOUT_MS / 1000.0
    last_audio_time = time.time()
    running = True

    while running:
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=silence_timeout)
            if chunk is None:
                running = False
            else:
                audio_buffer.extend(chunk)
                last_audio_time = time.time()
        except asyncio.TimeoutError:
            # Silence detected — process accumulated audio if we have enough
            elapsed = time.time() - last_audio_time
            if len(audio_buffer) > 1600 and elapsed >= silence_timeout:
                audio_data = bytes(audio_buffer)
                audio_buffer.clear()
                last_audio_time = time.time()

                logger.debug(
                    f"Silence detected, processing {len(audio_data)} bytes",
                    extra={"call_sid": call_sid},
                )

                loop = asyncio.get_event_loop()
                text, confidence = await loop.run_in_executor(
                    None, _run_stt_on_audio, audio_data, call_sid
                )

                if text and confidence >= STT_CONFIDENCE_THRESHOLD:
                    _store_transcript(call_sid, text, confidence)
                elif text:
                    logger.debug(
                        f"Low confidence rejected: '{text}' ({confidence:.2f})",
                        extra={"call_sid": call_sid},
                    )

    # Process any remaining audio at stream end
    if len(audio_buffer) > 1600:
        audio_data = bytes(audio_buffer)
        logger.debug(
            f"Stream ended, processing final {len(audio_data)} bytes",
            extra={"call_sid": call_sid},
        )
        loop = asyncio.get_event_loop()
        text, confidence = await loop.run_in_executor(
            None, _run_stt_on_audio, audio_data, call_sid
        )
        if text and confidence >= STT_CONFIDENCE_THRESHOLD:
            _store_transcript(call_sid, text, confidence)


# ── Twilio Media Stream WebSocket handler ────────────────────────────────────

async def handle_media_stream(websocket: WebSocket):
    """
    Handle a Twilio Media Stream WebSocket connection.

    Twilio sends JSON messages:
    - {"event": "connected"}
    - {"event": "start", "start": {"callSid": "...", "streamSid": "..."}}
    - {"event": "media", "media": {"payload": "<base64 audio>"}}
    - {"event": "stop"}
    """
    await websocket.accept()

    call_sid = ""
    stream_sid = ""
    audio_queue: asyncio.Queue = asyncio.Queue()
    stt_task: Optional[asyncio.Task] = None

    try:
        async for raw_message in websocket.iter_text():
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            event = msg.get("event", "")

            if event == "connected":
                logger.debug("Twilio Media Stream connected")

            elif event == "start":
                start_data = msg.get("start", {})
                call_sid = start_data.get("callSid", "")
                stream_sid = start_data.get("streamSid", "")
                logger.info(
                    f"Media stream started (stream={stream_sid[:16]}...)",
                    extra={"call_sid": call_sid},
                )
                stt_task = asyncio.create_task(
                    _process_audio_continuously(call_sid, audio_queue)
                )

            elif event == "media":
                payload = msg.get("media", {}).get("payload", "")
                if payload:
                    audio_bytes = base64.b64decode(payload)
                    await audio_queue.put(audio_bytes)

            elif event == "stop":
                logger.info("Media stream stopped", extra={"call_sid": call_sid})
                await audio_queue.put(None)
                break

    except WebSocketDisconnect:
        logger.info("Media stream WebSocket disconnected", extra={"call_sid": call_sid})
    except Exception as e:
        logger.error(f"Media stream error: {e}", extra={"call_sid": call_sid})
    finally:
        await audio_queue.put(None)
        if stt_task and not stt_task.done():
            try:
                await asyncio.wait_for(stt_task, timeout=10.0)
            except asyncio.TimeoutError:
                stt_task.cancel()
                logger.warning("STT task timed out, cancelled", extra={"call_sid": call_sid})
