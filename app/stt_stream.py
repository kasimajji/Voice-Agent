"""
Google STT v2 real-time streaming transcription via Twilio Media Streams.

Architecture:
1. Twilio sends real-time audio via WebSocket (base64 μ-law 8kHz)
2. Audio chunks are forwarded IMMEDIATELY to Google STT v2 streaming API
   via a thread-safe queue → generator pattern (true bidirectional streaming)
3. Google STT returns interim + final results in real-time as speech happens
4. Final transcripts are stored per-call and consumed by the state machine
5. Automatic reconnection handles Google's 5-minute streaming limit
6. Graceful fallback: if Google STT fails, Twilio's built-in STT still works

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
    STT_LANGUAGE_CODE,
)
from .logging_config import get_logger

logger = get_logger("stt_stream")

# ── Per-call transcript storage ──────────────────────────────────────────────
# Key: call_sid → dict with text, confidence, timestamp, is_final
_transcripts: dict[str, dict] = {}
_transcript_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
# Lock for thread-safe transcript updates (STT runs in background thread)
_transcript_lock = threading.Lock()


def get_transcript(call_sid: str, not_before: float = 0.0) -> Optional[dict]:
    """
    Get the latest transcript for a call (non-blocking, thread-safe).
    If not_before is set, only return transcripts that arrived AFTER that timestamp.
    This prevents stale transcripts from a previous turn overriding the current one.
    """
    with _transcript_lock:
        t = _transcripts.get(call_sid)
        if t and not_before > 0 and t.get("timestamp", 0) < not_before:
            logger.debug(
                f"Ignoring stale transcript (ts={t['timestamp']:.1f} < not_before={not_before:.1f})",
                extra={"call_sid": call_sid},
            )
            return None
        return t.copy() if t else None


async def wait_for_transcript(call_sid: str, timeout: float = 30.0) -> Optional[dict]:
    """Wait for a transcript to arrive for a call (async, with timeout)."""
    event = _transcript_events[call_sid]
    event.clear()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return get_transcript(call_sid)
    except asyncio.TimeoutError:
        logger.warning(f"Transcript wait timed out", extra={"call_sid": call_sid})
        return None


def clear_transcript(call_sid: str):
    """Clear transcript after it's been consumed by the state machine."""
    with _transcript_lock:
        _transcripts.pop(call_sid, None)
    if call_sid in _transcript_events:
        _transcript_events[call_sid].clear()


def _store_transcript(call_sid: str, text: str, confidence: float, is_final: bool = True):
    """Store a transcript and signal waiters. Thread-safe.
    
    When multiple FINAL results arrive in one turn (e.g., long utterance split
    by Google), keep the LONGEST one to avoid losing customer speech.
    """
    with _transcript_lock:
        existing = _transcripts.get(call_sid)
        # Only overwrite with final results, or if no existing final result
        if is_final or not existing or not existing.get("is_final"):
            # If both are final, keep the longer one (avoids losing speech when
            # Google sends multiple final results in one turn)
            if is_final and existing and existing.get("is_final"):
                if len(text) > len(existing["text"]):
                    _transcripts[call_sid] = {
                        "text": text,
                        "confidence": confidence,
                        "timestamp": time.time(),
                        "is_final": is_final,
                    }
            else:
                _transcripts[call_sid] = {
                    "text": text,
                    "confidence": confidence,
                    "timestamp": time.time(),
                    "is_final": is_final,
                }
    if is_final:
        _transcript_events[call_sid].set()
    if text and is_final:
        logger.info(
            f"[Google STT FINAL] '{text[:80]}' (conf={confidence:.2f})",
            extra={"call_sid": call_sid},
        )


# ── Google STT v2 real-time streaming recognizer ─────────────────────────────

# Google STT streaming has a 5-minute limit. We reconnect before that.
_STREAM_TIME_LIMIT_SEC = 240  # 4 minutes, reconnect before the 5-min hard limit
# Minimum audio bytes before we consider processing (avoid empty requests)
_MIN_AUDIO_BYTES = 160  # 20ms of 8kHz μ-law


def _build_recognition_config() -> speech.RecognitionConfig:
    """Build Google STT recognition config optimized for telephony audio."""
    return speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        language_code=STT_LANGUAGE_CODE,
        model="telephony",
        use_enhanced=True,
        enable_automatic_punctuation=True,
        # Enable word-level confidence for better accuracy assessment
        enable_word_confidence=True,
        # Max alternatives for better accuracy
        max_alternatives=1,
        speech_contexts=[
            speech.SpeechContext(
                phrases=[
                    # Appliances
                    "refrigerator", "washer", "dryer", "dishwasher", "oven",
                    "HVAC", "air conditioner", "furnace", "heat pump",
                    "microwave", "garbage disposal", "ice maker",
                    # Symptoms
                    "not cooling", "not heating", "leaking", "making noise",
                    "error code", "won't start", "won't turn on", "broken",
                    "vibrating", "overheating", "freezing",
                    # Actions
                    "schedule", "technician", "appointment", "troubleshoot",
                    "morning", "afternoon", "evening",
                    # Confirmations
                    "yes", "no", "correct", "wrong", "right", "okay",
                    # Email
                    "gmail", "yahoo", "outlook", "hotmail", "at", "dot com",
                    "dot net", "dot org",
                    # Names (common patterns)
                    "my name is", "I'm", "this is",
                ],
                boost=15.0,
            )
        ],
    )


def _build_streaming_config() -> speech.StreamingRecognitionConfig:
    """Build streaming config with interim results enabled for real-time feedback."""
    return speech.StreamingRecognitionConfig(
        config=_build_recognition_config(),
        interim_results=True,   # Get partial results in real-time
        single_utterance=False,  # Don't stop after first utterance
    )


class _AudioStream:
    """
    Thread-safe audio stream that bridges async WebSocket audio to
    synchronous Google STT streaming_recognize generator.
    
    Audio chunks are pushed from the async WebSocket handler and
    consumed by the Google STT background thread via the generator protocol.
    """
    
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._started_at = time.time()
    
    def push(self, audio_bytes: bytes):
        """Push audio chunk from async context. Non-blocking."""
        if not self._closed:
            self._queue.put(audio_bytes)
    
    def close(self):
        """Signal end of stream."""
        self._closed = True
        self._queue.put(None)  # Sentinel to unblock generator
    
    @property
    def is_expired(self) -> bool:
        """Check if stream has exceeded Google's time limit."""
        return (time.time() - self._started_at) > _STREAM_TIME_LIMIT_SEC
    
    def generator(self):
        """
        Yield StreamingRecognizeRequests for Google STT.
        Blocks on queue.get() until audio arrives or stream is closed.
        This runs in a background thread.
        
        Sends an initial silence frame so Google STT keeps the stream open
        even if real audio hasn't arrived yet (important after reconnect).
        """
        # Send 200ms of μ-law silence (0xFF = silence in μ-law) to bootstrap
        # the stream. Without this, Google STT closes immediately on reconnect
        # because no audio arrives before the internal timeout.
        yield speech.StreamingRecognizeRequest(
            audio_content=b'\xff' * 1600  # 200ms at 8kHz
        )
        
        while not self._closed:
            try:
                # Block up to 0.5s waiting for audio, then check if closed
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if chunk is None:
                # Sentinel — stream ended
                break
            
            if len(chunk) >= _MIN_AUDIO_BYTES:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)


def _run_streaming_stt(audio_stream: _AudioStream, call_sid: str):
    """
    Run Google STT streaming recognition in a background thread.
    
    This is synchronous because Google's streaming_recognize() is a
    blocking iterator. It processes responses as they arrive in real-time.
    
    Handles:
    - Interim results (stored as non-final for UI/logging)
    - Final results (stored and signaled to state machine)
    - Stream expiration (returns so caller can reconnect)
    - All errors (logged, never crashes)
    """
    try:
        client = speech.SpeechClient()
        streaming_config = _build_streaming_config()
        
        logger.info(
            "Google STT streaming session started",
            extra={"call_sid": call_sid},
        )
        
        requests = audio_stream.generator()
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )
        
        for response in responses:
            if audio_stream._closed:
                break
                
            for result in response.results:
                if not result.alternatives:
                    continue
                
                alt = result.alternatives[0]
                transcript = alt.transcript.strip()
                confidence = alt.confidence if alt.confidence else 0.0
                
                if not transcript:
                    continue
                
                if result.is_final:
                    # Final result — store with high priority
                    _store_transcript(call_sid, transcript, confidence, is_final=True)
                else:
                    # Interim result — store for potential use, don't signal
                    _store_transcript(call_sid, transcript, confidence, is_final=False)
                    logger.debug(
                        f"[Google STT interim] '{transcript[:60]}'",
                        extra={"call_sid": call_sid},
                    )
        
        logger.info(
            "Google STT streaming session ended normally",
            extra={"call_sid": call_sid},
        )
        
    except Exception as e:
        error_msg = str(e)
        # Don't log "cancelled" as errors — that's normal on call end
        if "cancelled" in error_msg.lower() or "deadline exceeded" in error_msg.lower():
            logger.debug(
                f"Google STT stream ended: {error_msg[:100]}",
                extra={"call_sid": call_sid},
            )
        else:
            logger.error(
                f"Google STT streaming error: {error_msg[:200]}",
                extra={"call_sid": call_sid},
            )


async def _manage_stt_sessions(call_sid: str, audio_queue: asyncio.Queue):
    """
    Manage Google STT streaming sessions for a call.
    
    Handles:
    - Starting the initial STT session
    - Auto-reconnecting when the 4-minute limit approaches
    - Forwarding audio from the async queue to the sync audio stream
    - Clean shutdown when the call ends
    """
    audio_stream: Optional[_AudioStream] = None
    stt_thread: Optional[threading.Thread] = None
    running = True
    
    def _start_new_session():
        """Start a new Google STT streaming session in a background thread."""
        nonlocal audio_stream, stt_thread
        
        # Close previous stream if any
        if audio_stream:
            audio_stream.close()
        
        audio_stream = _AudioStream(call_sid)
        stt_thread = threading.Thread(
            target=_run_streaming_stt,
            args=(audio_stream, call_sid),
            daemon=True,
            name=f"stt-{call_sid[:8]}",
        )
        stt_thread.start()
        logger.debug("Started new STT session thread", extra={"call_sid": call_sid})
    
    # Start first session
    _start_new_session()
    
    try:
        while running:
            try:
                # Wait for audio with timeout to check for stream expiration
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if we need to reconnect (approaching 5-min limit)
                if audio_stream and audio_stream.is_expired:
                    logger.info(
                        "STT stream approaching time limit, reconnecting...",
                        extra={"call_sid": call_sid},
                    )
                    _start_new_session()
                continue
            
            if chunk is None:
                # Call ended
                running = False
                break
            
            # Forward audio to Google STT immediately (real-time!)
            if audio_stream and not audio_stream._closed:
                audio_stream.push(chunk)
                
                # Check for stream expiration on each chunk
                if audio_stream.is_expired:
                    logger.info(
                        "STT stream time limit reached, reconnecting...",
                        extra={"call_sid": call_sid},
                    )
                    _start_new_session()
    
    except Exception as e:
        logger.error(f"STT session manager error: {e}", extra={"call_sid": call_sid})
    finally:
        # Clean shutdown
        if audio_stream:
            audio_stream.close()
        if stt_thread and stt_thread.is_alive():
            stt_thread.join(timeout=5.0)
            if stt_thread.is_alive():
                logger.warning("STT thread did not exit cleanly", extra={"call_sid": call_sid})
        logger.info("STT session manager stopped", extra={"call_sid": call_sid})


# ── Twilio Media Stream WebSocket handler ────────────────────────────────────

async def handle_media_stream(websocket: WebSocket):
    """
    Handle a Twilio Media Stream WebSocket connection.

    Twilio sends JSON messages:
    - {"event": "connected"}
    - {"event": "start", "start": {"callSid": "...", "streamSid": "..."}}
    - {"event": "media", "media": {"payload": "<base64 audio>"}}
    - {"event": "stop"}
    
    Audio is forwarded to Google STT in real-time (not batched).
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
                logger.debug("Twilio Media Stream WebSocket connected")

            elif event == "start":
                start_data = msg.get("start", {})
                call_sid = start_data.get("callSid", "")
                stream_sid = start_data.get("streamSid", "")
                logger.info(
                    f"Media stream started (stream={stream_sid[:16]}...)",
                    extra={"call_sid": call_sid},
                )
                # Start real-time STT session manager
                stt_task = asyncio.create_task(
                    _manage_stt_sessions(call_sid, audio_queue)
                )

            elif event == "media":
                payload = msg.get("media", {}).get("payload", "")
                if payload:
                    audio_bytes = base64.b64decode(payload)
                    await audio_queue.put(audio_bytes)

            elif event == "stop":
                logger.info(
                    f"Media stream stopped",
                    extra={"call_sid": call_sid},
                )
                await audio_queue.put(None)  # Signal end
                break

    except WebSocketDisconnect:
        logger.info("Media stream WebSocket disconnected", extra={"call_sid": call_sid})
    except Exception as e:
        logger.error(f"Media stream error: {e}", extra={"call_sid": call_sid})
    finally:
        # Ensure queue gets sentinel so STT manager shuts down
        try:
            audio_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        
        if stt_task and not stt_task.done():
            try:
                await asyncio.wait_for(stt_task, timeout=8.0)
            except asyncio.TimeoutError:
                stt_task.cancel()
                logger.warning("STT manager timed out on shutdown", extra={"call_sid": call_sid})
        
        # Clean up transcript storage for ended calls (after a delay to allow consumption)
        # Don't clean immediately — the state machine may still need the last transcript
        logger.info("Media stream handler finished", extra={"call_sid": call_sid})
