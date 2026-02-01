"""
Tier 3: Gemini Vision Analysis
Analyze appliance images using Gemini multimodal model.
"""
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path

from .config import GEMINI_API_KEY, GEMINI_MODEL


def analyze_image_with_gemini(
    image_path: str,
    appliance_type: Optional[str] = None,
    symptom_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze an appliance image using Gemini Vision.
    
    Args:
        image_path: Path to the image file
        appliance_type: Type of appliance (from call context)
        symptom_summary: Summary of reported symptoms (from call context)
    
    Returns:
        {
            "summary": "Description of what was found in the image",
            "troubleshooting": "Step-by-step troubleshooting suggestions"
        }
    """
    if not GEMINI_API_KEY:
        print("[Vision] No GEMINI_API_KEY set, using fallback response")
        return fallback_analysis(appliance_type, symptom_summary)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        image_data = load_image_as_base64(image_path)
        if not image_data:
            print(f"[Vision] Failed to load image: {image_path}")
            return fallback_analysis(appliance_type, symptom_summary)
        
        mime_type = get_mime_type(image_path)
        
        context_parts = []
        if appliance_type:
            context_parts.append(f"Appliance type: {appliance_type}")
        if symptom_summary:
            context_parts.append(f"Reported symptoms: {symptom_summary}")
        
        context = "\n".join(context_parts) if context_parts else "No additional context provided."
        
        prompt = f"""You are an expert appliance repair technician analyzing an image sent by a customer.

Context from the customer's call:
{context}

Analyze this image and provide:

1. **SUMMARY**: Describe what you observe in the image that is relevant to diagnosing the appliance issue. Look for:
   - Error codes or warning lights on displays
   - Visible damage, rust, or wear
   - Leaks, frost buildup, or condensation
   - Unusual positioning of parts
   - Model/serial number if visible

2. **TROUBLESHOOTING**: Provide 2-4 safe troubleshooting steps the customer can try at home. Be specific and practical. If the issue appears serious or requires professional repair, clearly state that.

Format your response as JSON:
{{
    "summary": "Your detailed observations here",
    "troubleshooting": "Step 1: ...\\nStep 2: ...\\nStep 3: ..."
}}

If you cannot identify relevant diagnostic information from the image, still provide general guidance based on the context provided."""

        model = genai.GenerativeModel(GEMINI_MODEL)
        
        response = model.generate_content([
            prompt,
            {
                "mime_type": mime_type,
                "data": image_data
            }
        ])
        
        result_text = response.text.strip()
        print(f"[Vision] Raw response: {result_text[:200]}...")
        
        return parse_vision_response(result_text, appliance_type, symptom_summary)
        
    except Exception as e:
        print(f"[Vision] Error during analysis: {e}")
        return fallback_analysis(appliance_type, symptom_summary)


def load_image_as_base64(image_path: str) -> Optional[str]:
    """Load an image file and return as base64 string."""
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[Vision] Error loading image: {e}")
        return None


def get_mime_type(image_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    return mime_map.get(ext, "image/jpeg")


def parse_vision_response(
    response_text: str,
    appliance_type: Optional[str],
    symptom_summary: Optional[str]
) -> Dict[str, Any]:
    """Parse the Gemini response, handling JSON or plain text."""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        result = json.loads(cleaned)
        return {
            "summary": result.get("summary", "Analysis complete."),
            "troubleshooting": result.get("troubleshooting", "")
        }
    except json.JSONDecodeError:
        print("[Vision] Failed to parse JSON, using raw text")
        
        lines = response_text.split("\n")
        summary_lines = []
        troubleshooting_lines = []
        current_section = "summary"
        
        for line in lines:
            line_lower = line.lower()
            if "troubleshoot" in line_lower or "steps" in line_lower:
                current_section = "troubleshooting"
                continue
            
            if current_section == "summary":
                summary_lines.append(line)
            else:
                troubleshooting_lines.append(line)
        
        return {
            "summary": "\n".join(summary_lines).strip() or response_text[:500],
            "troubleshooting": "\n".join(troubleshooting_lines).strip()
        }


def fallback_analysis(
    appliance_type: Optional[str],
    symptom_summary: Optional[str]
) -> Dict[str, Any]:
    """Provide a fallback response when vision analysis is unavailable."""
    appliance = appliance_type or "appliance"
    
    summary = f"Image received for {appliance} diagnosis."
    if symptom_summary:
        summary += f" Based on your reported issue ({symptom_summary}), a technician review is recommended."
    else:
        summary += " Our team will review this image to assist with diagnosis."
    
    troubleshooting = f"""Check that the {appliance} is properly plugged in and receiving power
Inspect for any visible damage, leaks, or unusual sounds
Try unplugging the {appliance} for 60 seconds then plugging it back in
If the issue persists, a technician visit may be necessary"""
    
    return {
        "summary": summary,
        "troubleshooting": troubleshooting
    }
