from huggingface_hub import login
import os
import sys

# Get token from environment variable for security
token = os.getenv("HF_TOKEN")

if not token:
    print("Error: HF_TOKEN environment variable not set.")
    print("Please run: $env:HF_TOKEN='your_token_here' in PowerShell before running this script.")
    sys.exit(1)

try:
    login(token=token, add_to_git_credential=True)
    print("--- Success: Celeste-Imperia Authenticated ---")
except Exception as e:
    print(f"Login Failed: {e}")
    sys.exit(1)