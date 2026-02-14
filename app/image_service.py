"""
Tier 3: Image Upload Service
Token creation, URL building, and email sending for vision-based diagnosis.
"""
import os
import uuid
import re
from datetime import datetime, timedelta
from typing import Optional

from .config import APP_BASE_URL
from .db import SessionLocal
from .models import ImageUploadToken
from .logging_config import get_logger

logger = get_logger("image_service")


def generate_upload_token() -> str:
    """Generate a secure random token for image uploads."""
    return uuid.uuid4().hex


def build_upload_url(token: str) -> str:
    """Build the full upload URL using APP_BASE_URL."""
    return f"{APP_BASE_URL}/upload/{token}"


def create_image_upload_token(
    call_sid: str,
    email: str,
    appliance_type: Optional[str] = None,
    symptom_summary: Optional[str] = None,
    expiration_hours: int = 24
) -> ImageUploadToken:
    """
    Create and persist an image upload token.
    
    Args:
        call_sid: The Twilio call SID
        email: Customer's email address
        appliance_type: Type of appliance (from call)
        symptom_summary: Summary of symptoms (from call)
        expiration_hours: Hours until token expires (default 24)
    
    Returns:
        The created ImageUploadToken record
    """
    token = generate_upload_token()
    now = datetime.utcnow()
    expires_at = now + timedelta(hours=expiration_hours)
    
    db = SessionLocal()
    try:
        upload_token = ImageUploadToken(
            token=token,
            call_sid=call_sid,
            email=email,
            appliance_type=appliance_type,
            symptom_summary=symptom_summary,
            created_at=now,
            expires_at=expires_at
        )
        db.add(upload_token)
        db.commit()
        db.refresh(upload_token)
        
        logger.info(f"Created upload token for CallSid: {call_sid}, Email: {email}")
        return upload_token
    finally:
        db.close()


def get_upload_token(token: str) -> Optional[ImageUploadToken]:
    """Retrieve an upload token by its token string."""
    db = SessionLocal()
    try:
        return db.query(ImageUploadToken).filter(
            ImageUploadToken.token == token
        ).first()
    finally:
        db.close()


def is_token_valid(upload_token: ImageUploadToken) -> bool:
    """Check if a token is valid (not expired and not used)."""
    if upload_token is None:
        return False
    now = datetime.utcnow()
    return upload_token.expires_at > now and upload_token.used_at is None


def mark_token_used(token: str, image_url: str) -> Optional[ImageUploadToken]:
    """Mark a token as used and store the image URL."""
    db = SessionLocal()
    try:
        upload_token = db.query(ImageUploadToken).filter(
            ImageUploadToken.token == token
        ).first()
        
        if upload_token:
            upload_token.used_at = datetime.utcnow()
            upload_token.image_url = image_url
            db.commit()
            db.refresh(upload_token)
        
        return upload_token
    finally:
        db.close()


def update_token_analysis(
    token: str,
    analysis_summary: str,
    troubleshooting_tips: str,
    is_appliance_image: bool = True
) -> Optional[ImageUploadToken]:
    """Update the token with vision analysis results."""
    db = SessionLocal()
    try:
        upload_token = db.query(ImageUploadToken).filter(
            ImageUploadToken.token == token
        ).first()
        
        if upload_token:
            upload_token.analysis_summary = analysis_summary
            upload_token.troubleshooting_tips = troubleshooting_tips
            upload_token.is_appliance_image = is_appliance_image
            db.commit()
            db.refresh(upload_token)
        
        return upload_token
    finally:
        db.close()


def reset_upload_for_reupload(call_sid: str) -> Optional[str]:
    """
    Reset the most recent upload token for a call so the customer can re-upload.
    Clears analysis fields and used_at so the token can be reused.
    Returns the upload URL if successful, None otherwise.
    """
    db = SessionLocal()
    try:
        upload_token = db.query(ImageUploadToken).filter(
            ImageUploadToken.call_sid == call_sid
        ).order_by(ImageUploadToken.created_at.desc()).first()
        
        if upload_token:
            upload_token.used_at = None
            upload_token.image_url = None
            upload_token.analysis_summary = None
            upload_token.troubleshooting_tips = None
            upload_token.is_appliance_image = None
            db.commit()
            db.refresh(upload_token)
            logger.info(f"Reset upload token for re-upload: {upload_token.token[:8]}...", 
                       extra={"call_sid": call_sid})
            return build_upload_url(upload_token.token)
        return None
    finally:
        db.close()


def get_upload_status_by_call_sid(call_sid: str) -> Optional[dict]:
    """
    Check if an image has been uploaded for a given call.
    Used by voice flow to poll for upload completion.
    
    Returns:
        dict with upload status and analysis if available, or None if no token exists
    """
    db = SessionLocal()
    try:
        # Get the most recent token for this call
        upload_token = db.query(ImageUploadToken).filter(
            ImageUploadToken.call_sid == call_sid
        ).order_by(ImageUploadToken.created_at.desc()).first()
        
        if not upload_token:
            return None
        
        return {
            "token": upload_token.token,
            "email": upload_token.email,
            "image_uploaded": upload_token.used_at is not None,
            "analysis_ready": upload_token.analysis_summary is not None,
            "analysis_summary": upload_token.analysis_summary,
            "troubleshooting_tips": upload_token.troubleshooting_tips,
            "appliance_type": upload_token.appliance_type,
            "is_appliance_image": upload_token.is_appliance_image
        }
    finally:
        db.close()


def validate_email(email: str) -> bool:
    """Basic email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def send_upload_email(email: str, upload_url: str, appliance_type: Optional[str] = None) -> bool:
    """
    Send an email with the image upload link.
    
    In dev mode (no SENDGRID_API_KEY), logs to console instead.
    
    Args:
        email: Recipient email address
        upload_url: The full upload URL
        appliance_type: Optional appliance type for personalization
    
    Returns:
        True if email was sent (or logged), False on error
    """
    sendgrid_key = os.getenv("SENDGRID_API_KEY")
    
    appliance_text = f" for your {appliance_type}" if appliance_type else ""
    
    subject = "Sears Home Services - Upload Photo for Diagnosis"
    body = f"""Hello,

Thank you for calling Sears Home Services. To help us better diagnose the issue{appliance_text}, please upload a photo of your appliance showing the problem area.

Click the link below to upload your photo:
{upload_url}

This link will expire in 24 hours.

Tips for a helpful photo:
- Show any error codes or warning lights on the display
- Capture any visible damage, leaks, or frost buildup
- Include the model number label if possible

After you upload, our system will analyze the image and provide additional troubleshooting suggestions.

Thank you,
Sears Home Services Team
"""
    
    if not sendgrid_key:
        logger.info(f"[DEV MODE] Email would be sent to: {email}")
        logger.info(f"[DEV MODE] 📎 UPLOAD LINK: {upload_url}")
        logger.debug(f"Subject: {subject}")
        logger.debug(f"Body: {body[:200]}...")
        return True
    
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
        message = Mail(
            from_email=os.getenv("SENDGRID_FROM_EMAIL", "noreply@searshomeservices.com"),
            to_emails=email,
            subject=subject,
            plain_text_content=body
        )
        
        sg = SendGridAPIClient(sendgrid_key)
        response = sg.send(message)
        
        logger.info(f"Email sent to {email}, status: {response.status_code}")
        return response.status_code in [200, 201, 202]
        
    except Exception as e:
        logger.error(f"Email error: {e}")
        logger.warning(f"Falling back to console output for: {email}")
        logger.info(f"Upload URL: {upload_url}")
        return True
