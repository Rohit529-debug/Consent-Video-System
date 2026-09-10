import requests
from app.config import ELEVENLABS_API_KEY, ELEVENLABS_STT_MODEL, ELEVENLABS_STT_URL


# send audio file to ElevenLabs Scribe and return transcript + word timings
def transcribe_audio(audio_path: str) -> dict:
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    data = {
        "model_id": ELEVENLABS_STT_MODEL,
        "tag_audio_events": "false",
        "diarize": "true",  # helps flag a second speaker for prompting checks
    }

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/wav")}
        response = requests.post(
            ELEVENLABS_STT_URL, headers=headers, data=data, files=files, timeout=120
        )

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs STT failed ({response.status_code}): {response.text}")

    raw = response.json()
    return parse_stt_response(raw)


# normalize the ElevenLabs response into a simpler shape the rest of the app uses
def parse_stt_response(raw: dict) -> dict:
    transcript_text = raw.get("text", "")
    detected_language = raw.get("language_code") or raw.get("language") or "unknown"

    words = []
    for w in raw.get("words", []):
        words.append({
            "text": w.get("text", ""),
            "start": w.get("start"),
            "end": w.get("end"),
            "speaker": w.get("speaker_id"),
        })

    speaker_ids = {w["speaker"] for w in words if w.get("speaker") is not None}

    return {
        "transcript": transcript_text,
        "language": detected_language,
        "words": words,
        "distinct_speakers_detected": len(speaker_ids) if speaker_ids else None,
    }
