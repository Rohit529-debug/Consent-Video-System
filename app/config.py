import os
from dotenv import load_dotenv

# load .env file from project root
load_dotenv()

# ElevenLabs Scribe STT settings
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# temp working directory for uploaded videos / extracted audio
TMP_DIR = os.getenv("TMP_DIR", "./tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# fail fast if keys are missing so errors are obvious at startup, not mid-request
def check_keys():
    missing = []
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required .env keys: {', '.join(missing)}")
