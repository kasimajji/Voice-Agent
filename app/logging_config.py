"""
Production-grade logging configuration for Voice Agent.

Features:
- Structured JSON logging for production (log aggregators like Datadog, CloudWatch, ELK)
- Colored console output for development
- Call context tracking (call_sid, step, speaker)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

Usage:
    from .logging_config import get_logger, log_conversation, log_error
    
    logger = get_logger("twilio")
    logger.info("Message", extra={"call_sid": call_sid})
    
    log_conversation(call_sid, "AGENT", "Hello!", step="greet")
"""

import logging
import sys
import json
import os
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Structured JSON logging for production (easy to parse by log aggregators)."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present (call_sid, step, speaker, etc.)
        if hasattr(record, "call_sid") and record.call_sid:
            log_data["call_sid"] = record.call_sid
        if hasattr(record, "step") and record.step:
            log_data["step"] = record.step
        if hasattr(record, "speaker") and record.speaker:
            log_data["speaker"] = record.speaker
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data") and record.extra_data:
            log_data["data"] = record.extra_data
            
        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable colored console output for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Speaker-specific colors for conversation logs
    SPEAKER_COLORS = {
        "AGENT": "\033[34m",     # Blue
        "CUSTOMER": "\033[33m",  # Yellow
        "SYSTEM": "\033[36m",    # Cyan
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Build context string from extra fields
        context_parts = []
        call_sid = getattr(record, "call_sid", None)
        step = getattr(record, "step", None)
        speaker = getattr(record, "speaker", None)
        
        if call_sid:
            # Show truncated call_sid for readability
            short_sid = call_sid[:12] + "..." if len(call_sid) > 12 else call_sid
            context_parts.append(f"Call:{short_sid}")
        if step:
            context_parts.append(f"Step:{step}")
        
        context = f" [{', '.join(context_parts)}]" if context_parts else ""
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Special formatting for conversation logs
        if speaker:
            speaker_color = self.SPEAKER_COLORS.get(speaker, self.RESET)
            return (
                f"{color}[{timestamp}]{self.RESET}{context} "
                f"{speaker_color}{self.BOLD}[{speaker}]{self.RESET} {record.getMessage()}"
            )
        
        return f"{color}[{timestamp}] {record.levelname:8}{self.RESET}{context} {record.getMessage()}"


class CallContextFilter(logging.Filter):
    """Filter that adds default values for call context fields."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Ensure all context fields exist (even if empty)
        if not hasattr(record, "call_sid"):
            record.call_sid = ""
        if not hasattr(record, "step"):
            record.step = ""
        if not hasattr(record, "speaker"):
            record.speaker = ""
        if not hasattr(record, "extra_data"):
            record.extra_data = None
        return True


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True for production, False for dev)
        log_file: Optional file path to write logs
    
    Returns:
        Configured root logger for the app
    """
    # Create app logger
    logger = logging.getLogger("voice_agent")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()  # Remove any existing handlers
    
    # Add context filter
    context_filter = CallContextFilter()
    logger.addFilter(context_filter)
    
    # Choose formatter based on environment
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = ConsoleFormatter()
    
    # Console handler (always enabled, writes to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    console_handler.stream = sys.stdout  # Ensure stdout
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Initialize logging from environment variables
_LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
_LOG_FORMAT_JSON = os.getenv("LOG_FORMAT", "console").lower() == "json"
_LOG_FILE = os.getenv("LOG_FILE", None)

# Create default logger instance
_root_logger = setup_logging(
    log_level=_LOG_LEVEL,
    json_format=_LOG_FORMAT_JSON,
    log_file=_LOG_FILE
)


def get_logger(name: str = "") -> logging.Logger:
    """
    Get a child logger with the given name.
    
    Args:
        name: Logger name suffix (e.g., "twilio", "llm", "db")
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"voice_agent.{name}")
    return logging.getLogger("voice_agent")


# =============================================================================
# Convenience functions for structured logging
# =============================================================================

def log_conversation(call_sid: str, speaker: str, message: str, step: str = ""):
    """
    Log conversation turns (agent/customer speech).
    
    Args:
        call_sid: Twilio call SID
        speaker: AGENT, CUSTOMER, or SYSTEM
        message: The spoken text
        step: Current conversation step
    """
    logger = get_logger("conversation")
    logger.info(
        message,
        extra={"call_sid": call_sid, "speaker": speaker, "step": step}
    )


def log_state_change(call_sid: str, from_step: str, to_step: str, **extra_data):
    """
    Log state machine transitions.
    
    Args:
        call_sid: Twilio call SID
        from_step: Previous step
        to_step: New step
        **extra_data: Additional context data
    """
    logger = get_logger("state")
    logger.debug(
        f"State: {from_step} → {to_step}",
        extra={"call_sid": call_sid, "step": to_step, "extra_data": extra_data}
    )


def log_call_start(call_sid: str, from_number: str, to_number: str):
    """Log incoming call start."""
    logger = get_logger("call")
    logger.info(
        f"📞 INCOMING CALL from {from_number} to {to_number}",
        extra={"call_sid": call_sid, "step": "start"}
    )


def log_call_end(call_sid: str, resolved: bool = False, reason: str = ""):
    """Log call end."""
    logger = get_logger("call")
    status = "✅ RESOLVED" if resolved else "📱 ENDED"
    msg = f"{status}" + (f" - {reason}" if reason else "")
    logger.info(msg, extra={"call_sid": call_sid, "step": "end"})


def log_error(call_sid: str, error: Exception, step: str = "", context: str = ""):
    """
    Log errors with full context.
    
    Args:
        call_sid: Twilio call SID
        error: The exception
        step: Current conversation step
        context: Additional context about what was happening
    """
    logger = get_logger("error")
    msg = f"❌ {context}: {type(error).__name__}: {error}" if context else f"❌ {type(error).__name__}: {error}"
    logger.error(msg, extra={"call_sid": call_sid, "step": step}, exc_info=True)


def log_llm_call(call_sid: str, function: str, input_text: str, output: str, duration_ms: int = 0):
    """Log LLM API calls for debugging."""
    logger = get_logger("llm")
    logger.debug(
        f"🤖 {function}: '{input_text[:50]}...' → '{output[:50]}...' ({duration_ms}ms)",
        extra={"call_sid": call_sid, "duration_ms": duration_ms}
    )


def log_db_operation(operation: str, table: str, success: bool = True, **extra_data):
    """Log database operations."""
    logger = get_logger("db")
    status = "✓" if success else "✗"
    logger.debug(
        f"🗄️ {status} {operation} on {table}",
        extra={"extra_data": extra_data}
    )


def log_external_service(service: str, operation: str, success: bool = True, **extra_data):
    """Log external service calls (Twilio, SendGrid, etc.)."""
    logger = get_logger("external")
    status = "✓" if success else "✗"
    logger.info(
        f"🌐 {status} {service}: {operation}",
        extra={"extra_data": extra_data}
    )
