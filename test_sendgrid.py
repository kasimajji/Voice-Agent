"""Quick test for SendGrid integration."""
import os
import sys

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv('app/.env')

api_key = os.getenv('SENDGRID_API_KEY')
from_email = os.getenv('SENDGRID_FROM_EMAIL')

print(f"API Key: {api_key[:15]}..." if api_key else "API Key: NOT SET ❌")
print(f"From Email: {from_email}" if from_email else "From Email: NOT SET ❌")

if not api_key or not from_email:
    print("\n⚠️  Add these to app/.env:")
    print("SENDGRID_API_KEY=SG.xxxxx")
    print("SENDGRID_FROM_EMAIL=your@email.com")
    sys.exit(1)

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Send test email
test_email = input(f"\nSend test email to [{from_email}]: ").strip() or from_email

message = Mail(
    from_email=from_email,
    to_emails=test_email,
    subject='SendGrid Test - Sears Home Services',
    plain_text_content='This is a test email from your Voice AI agent. If you receive this, SendGrid is working!'
)

print(f"\nSending to {test_email}...")

try:
    sg = SendGridAPIClient(api_key)
    response = sg.send(message)
    print(f"Status: {response.status_code}")
    if response.status_code in [200, 201, 202]:
        print("✅ SUCCESS! Check your inbox.")
    else:
        print("❌ FAILED - check your API key and from_email")
except Exception as e:
    print(f"❌ Error: {e}")
