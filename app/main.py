import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import check_keys
from app.media_utils import save_upload, extract_audio, cleanup_files, validate_upload, InvalidUploadError
from app.stt import transcribe_audio
from app.statement_matcher import evaluate_statement
from app.video_analysis import analyze_video
from app.decision import build_final_decision, build_json_result, build_human_report, fuse_other_person_signal

app = FastAPI(title="AI Video Consent Verification System")


# quick health/startup check that required API keys are present
@app.on_event("startup")
def startup_check():
    check_keys()


@app.get("/health")
def health():
    return {"status": "ok"}


# serve the single-page frontend from the same origin as the API, so the browser
# needs no CORS config and the deployed app is one URL
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# main endpoint: upload a video + expected name, get back the full consent assessment
@app.post("/analyze-consent-video")
async def analyze_consent_video(
    video: UploadFile = File(...),
    expected_name: str = Form(...),
    language_hint: str = Form(default="auto"),  # "english" | "hindi" | "auto"
):
    video_path = None
    audio_path = None
    try:
        # step 0: validate upload before writing anything to disk
        video_bytes = await video.read()
        try:
            validate_upload(video_bytes, video.content_type)
        except InvalidUploadError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # step 1: persist upload
        video_path = save_upload(video_bytes, video.filename)

        # step 2: extract audio for STT
        audio_path = extract_audio(video_path)

        # step 3: transcribe with ElevenLabs Scribe (blocking HTTP call - run off the event loop)
        stt_result = await run_in_threadpool(transcribe_audio, audio_path)

        # step 4: resolve language (use STT-detected language unless caller pinned one)
        language = resolve_language(language_hint, stt_result["language"])

        # step 5: deterministic statement/name/order verification (with word timestamps)
        statement_result = evaluate_statement(stt_result["transcript"], expected_name, language, stt_result["words"])

        # step 6: multimodal video analysis via Gemini (blocking upload + poll loop)
        video_result = await run_in_threadpool(analyze_video, video_path)

        # step 7: fuse audio diarization with Gemini's other-person read, then decide
        other_person_fusion = fuse_other_person_signal(stt_result, video_result)
        final = build_final_decision(statement_result, video_result, other_person_fusion)

        # step 8: assemble outputs
        json_result = build_json_result(expected_name, stt_result, statement_result, video_result, final, other_person_fusion)
        human_report = build_human_report(json_result)

        return JSONResponse({
            "result": json_result,
            "human_readable_report": human_report,
        })

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        cleanup_files(video_path, audio_path)


# map STT language codes to our "english"/"hindi" template keys
def resolve_language(language_hint: str, detected_code: str) -> str:
    if language_hint in ("english", "hindi"):
        return language_hint
    detected_code = (detected_code or "").lower()
    if detected_code.startswith("hi"):
        return "hindi"
    return "english"
