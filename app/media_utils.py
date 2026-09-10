import subprocess
import os
import uuid
from app.config import TMP_DIR

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # ~200MB upload cap


# raised for bad uploads (too large / wrong content type) so the route can 400 cleanly
class InvalidUploadError(Exception):
    pass


# reject oversized or non-video uploads before anything touches disk
def validate_upload(file_bytes: bytes, content_type: str):
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidUploadError(f"Upload exceeds max size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    if not content_type or not content_type.startswith("video/"):
        raise InvalidUploadError(f"Unsupported content type: {content_type}")


# save an uploaded file's raw bytes to a temp path and return that path
def save_upload(file_bytes: bytes, original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1] or ".mp4"
    path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


# extract mono 16kHz wav audio from a video file, required for clean STT input
def extract_audio(video_path: str) -> str:
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(audio_path):
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return audio_path


# clean up temp files after a request finishes
def cleanup_files(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
