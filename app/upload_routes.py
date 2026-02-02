"""
Tier 3: Image Upload Routes
GET and POST handlers for the image upload flow.
"""
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from html import escape as html_escape
from .image_service import (
    get_upload_token,
    is_token_valid,
    mark_token_used,
    update_token_analysis
)
from .vision import analyze_image_with_gemini

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def ensure_upload_dir():
    """Ensure the uploads directory exists."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/upload/{token}", response_class=HTMLResponse)
async def upload_form(token: str):
    """
    GET /upload/{token}
    Display the image upload form if token is valid.
    """
    upload_token = get_upload_token(token)
    
    if not upload_token:
        return HTMLResponse(content=error_page("Invalid Link", 
            "This upload link is invalid or does not exist."), status_code=404)
    
    if not is_token_valid(upload_token):
        if upload_token.used_at:
            return HTMLResponse(content=error_page("Already Used",
                "This upload link has already been used. Check your email for the analysis results."), status_code=410)
        else:
            return HTMLResponse(content=error_page("Link Expired",
                "This upload link has expired. Please call us again to get a new link."), status_code=410)
    
    appliance_text = f" for your {html_escape(upload_token.appliance_type)}" if upload_token.appliance_type else ""
    
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Photo - Sears Home Services</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
        }}
        .logo {{
            text-align: center;
            margin-bottom: 24px;
        }}
        .logo h1 {{
            color: #1a1a2e;
            font-size: 24px;
            font-weight: 700;
        }}
        .logo span {{
            color: #667eea;
        }}
        h2 {{
            color: #333;
            font-size: 20px;
            margin-bottom: 16px;
            text-align: center;
        }}
        .info {{
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 16px;
            margin-bottom: 24px;
            border-radius: 0 8px 8px 0;
        }}
        .info p {{
            color: #4a5568;
            font-size: 14px;
            line-height: 1.6;
        }}
        .tips {{
            margin-bottom: 24px;
        }}
        .tips h3 {{
            color: #333;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .tips ul {{
            list-style: none;
            padding: 0;
        }}
        .tips li {{
            color: #666;
            font-size: 13px;
            padding: 6px 0;
            padding-left: 24px;
            position: relative;
        }}
        .tips li::before {{
            content: "✓";
            color: #667eea;
            position: absolute;
            left: 0;
        }}
        form {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .file-input {{
            border: 2px dashed #ddd;
            border-radius: 12px;
            padding: 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .file-input:hover {{
            border-color: #667eea;
            background: #f8f9ff;
        }}
        .file-input input {{
            display: none;
        }}
        .file-input label {{
            cursor: pointer;
            color: #666;
        }}
        .file-input .icon {{
            font-size: 48px;
            margin-bottom: 8px;
        }}
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }}
        button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}
        #fileName {{
            color: #667eea;
            font-weight: 500;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">
            <h1><span>Sears</span> Home Services</h1>
        </div>
        
        <h2>Upload a Photo{appliance_text}</h2>
        
        <div class="info">
            <p>Upload a clear photo of your appliance to help us diagnose the issue. 
               Our AI will analyze the image and provide troubleshooting suggestions.</p>
        </div>
        
        <div class="tips">
            <h3>For best results:</h3>
            <ul>
                <li>Show any error codes or warning lights</li>
                <li>Capture visible damage, leaks, or frost</li>
                <li>Include the model number if visible</li>
                <li>Use good lighting</li>
            </ul>
        </div>
        
        <form action="/upload/{token}" method="post" enctype="multipart/form-data">
            <div class="file-input" onclick="document.getElementById('imageFile').click()">
                <div class="icon">📷</div>
                <label for="imageFile">Click or tap to select a photo</label>
                <input type="file" id="imageFile" name="image" accept="image/jpeg,image/png,image/webp" required 
                       onchange="document.getElementById('fileName').textContent = this.files[0]?.name || ''">
                <div id="fileName"></div>
            </div>
            <button type="submit">Upload & Analyze</button>
        </form>
    </div>
</body>
</html>
""")


@router.post("/upload/{token}", response_class=HTMLResponse)
async def upload_image(token: str, image: UploadFile = File(...)):
    """
    POST /upload/{token}
    Handle image upload, save file, run vision analysis.
    """
    upload_token = get_upload_token(token)
    
    if not upload_token:
        return HTMLResponse(content=error_page("Invalid Link",
            "This upload link is invalid."), status_code=404)
    
    if not is_token_valid(upload_token):
        return HTMLResponse(content=error_page("Link Expired or Used",
            "This upload link has expired or already been used."), status_code=410)
    
    if image.content_type not in ALLOWED_MIME_TYPES:
        return HTMLResponse(content=error_page("Invalid File Type",
            f"Please upload a JPEG, PNG, or WebP image. You uploaded: {image.content_type}"), status_code=400)
    
    ensure_upload_dir()
    
    ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
    filename = f"{token}.{ext}"
    file_path = UPLOAD_DIR / filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"[Tier 3] Image saved: {file_path}")
        
        mark_token_used(token, str(file_path))
        
        analysis = analyze_image_with_gemini(
            image_path=str(file_path),
            appliance_type=upload_token.appliance_type,
            symptom_summary=upload_token.symptom_summary
        )
        
        # Check if it's actually an appliance image
        is_appliance = analysis.get("is_appliance_image", True)
        
        update_token_analysis(
            token=token,
            analysis_summary=analysis.get("summary", ""),
            troubleshooting_tips=analysis.get("troubleshooting", ""),
            is_appliance_image=is_appliance
        )
        
        # ISSUE 2.4: Show different page if not an appliance image
        if not is_appliance:
            return HTMLResponse(content=not_appliance_page(
                appliance_type=upload_token.appliance_type,
                summary=analysis.get("summary", "")
            ))
        
        return HTMLResponse(content=success_page(
            appliance_type=upload_token.appliance_type,
            summary=analysis.get("summary", "No analysis available"),
            troubleshooting=analysis.get("troubleshooting", "")
        ))
        
    except Exception as e:
        print(f"[Tier 3] Upload error: {e}")
        return HTMLResponse(content=error_page("Upload Failed",
            "There was an error processing your image. Please try again."), status_code=500)


def error_page(title: str, message: str) -> str:
    """Generate an error HTML page."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Sears Home Services</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
            text-align: center;
        }}
        .icon {{ font-size: 64px; margin-bottom: 16px; }}
        h1 {{ color: #e74c3c; font-size: 24px; margin-bottom: 16px; }}
        p {{ color: #666; line-height: 1.6; }}
        .contact {{
            margin-top: 24px;
            padding-top: 24px;
            border-top: 1px solid #eee;
            font-size: 14px;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">⚠️</div>
        <h1>{title}</h1>
        <p>{message}</p>
        <div class="contact">
            Need help? Call Sears Home Services
        </div>
    </div>
</body>
</html>
"""


def not_appliance_page(appliance_type: str, summary: str) -> str:
    """
    ISSUE 2.4: Generate a page for when uploaded image is NOT an appliance.
    Prompts user to upload a correct image.
    """
    appliance_text = html_escape(appliance_type) if appliance_type else "appliance"
    summary = html_escape(summary) if summary else ""
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Please Upload Appliance Photo - Sears Home Services</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
            text-align: center;
        }}
        .icon {{ font-size: 64px; margin-bottom: 16px; }}
        h1 {{ color: #f5576c; font-size: 24px; margin-bottom: 16px; }}
        p {{ color: #666; line-height: 1.6; margin-bottom: 16px; }}
        .what-we-saw {{
            background: #fff5f5;
            border-radius: 8px;
            padding: 16px;
            margin: 20px 0;
            text-align: left;
        }}
        .what-we-saw h3 {{
            color: #f5576c;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .what-we-saw p {{
            font-size: 14px;
            color: #666;
            margin: 0;
        }}
        .tips {{
            background: #f0f4ff;
            border-radius: 8px;
            padding: 16px;
            margin: 20px 0;
            text-align: left;
        }}
        .tips h3 {{
            color: #667eea;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .tips ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .tips li {{
            font-size: 13px;
            color: #666;
            padding: 4px 0;
        }}
        .tips li::before {{
            content: "✓ ";
            color: #667eea;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">📷</div>
        <h1>Please Upload a Photo of Your {appliance_text.title()}</h1>
        <p>The image you uploaded doesn't appear to show a home appliance.</p>
        
        <div class="what-we-saw">
            <h3>What we saw:</h3>
            <p>{summary}</p>
        </div>
        
        <div class="tips">
            <h3>For best results, please upload a photo that shows:</h3>
            <ul>
                <li>The {appliance_text} itself</li>
                <li>Any error codes or warning lights on the display</li>
                <li>Visible damage, leaks, or frost buildup</li>
                <li>The model number label if possible</li>
            </ul>
        </div>
        
        <p>You can use the same link to upload another photo.</p>
        <a href="javascript:history.back()" class="btn">← Try Again</a>
    </div>
</body>
</html>
"""


def success_page(appliance_type: str, summary: str, troubleshooting: str) -> str:
    """Generate a success HTML page with analysis results."""
    appliance_text = f" - {html_escape(appliance_type.title())}" if appliance_type else ""
    summary = html_escape(summary) if summary else ""
    troubleshooting = html_escape(troubleshooting) if troubleshooting else ""
    
    troubleshooting_html = ""
    if troubleshooting:
        steps = troubleshooting.split("\n")
        troubleshooting_html = "<ul>"
        for step in steps:
            step = step.strip()
            if step:
                troubleshooting_html += f"<li>{step}</li>"
        troubleshooting_html += "</ul>"
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Complete - Sears Home Services</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 32px;
        }}
        .icon {{ font-size: 64px; margin-bottom: 16px; }}
        h1 {{ color: #11998e; font-size: 24px; margin-bottom: 8px; }}
        .subtitle {{ color: #888; font-size: 14px; }}
        .section {{
            margin-bottom: 24px;
        }}
        .section h2 {{
            color: #333;
            font-size: 16px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #11998e;
        }}
        .analysis {{
            background: #f0fff4;
            border-radius: 8px;
            padding: 16px;
            color: #2d3748;
            line-height: 1.6;
        }}
        .troubleshooting ul {{
            list-style: none;
            padding: 0;
        }}
        .troubleshooting li {{
            background: #f8f9fa;
            margin-bottom: 8px;
            padding: 12px 16px;
            border-radius: 8px;
            color: #4a5568;
            position: relative;
            padding-left: 40px;
        }}
        .troubleshooting li::before {{
            content: "→";
            color: #11998e;
            font-weight: bold;
            position: absolute;
            left: 16px;
        }}
        .footer {{
            text-align: center;
            margin-top: 32px;
            padding-top: 24px;
            border-top: 1px solid #eee;
        }}
        .footer p {{
            color: #888;
            font-size: 14px;
            margin-bottom: 8px;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">✅</div>
            <h1>Analysis Complete{appliance_text}</h1>
            <p class="subtitle">Our AI has analyzed your image</p>
        </div>
        
        <div class="section">
            <h2>🔍 What We Found</h2>
            <div class="analysis">{summary}</div>
        </div>
        
        <div class="section troubleshooting">
            <h2>🔧 Suggested Troubleshooting Steps</h2>
            {troubleshooting_html if troubleshooting_html else '<p style="color: #666;">No specific steps identified. A technician visit may be needed.</p>'}
        </div>
        
        <div class="footer">
            <p>Still having issues?</p>
            <a href="tel:+18001234567" class="btn">📞 Call for Support</a>
        </div>
    </div>
</body>
</html>
"""
