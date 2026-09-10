import json
import time
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini is asked to return ONLY this JSON shape - no prose, no markdown fences.
# This covers everything text/audio-transcript alone can't answer:
# prompting, other-person presence, environment, socioeconomic read, video quality.
#
# The schema is deliberately chain-of-thought shaped: `observations` comes FIRST and
# forces the model to record raw, uninterpreted timestamped observations before any
# verdict field. Verdicts then have to cite those observation ids. Without this, the
# model jumped straight to a verdict and left the evidence arrays empty (an
# INDETERMINATE with no evidence is useless to a human reviewer), or justified a
# verdict with an observation it never actually established (a blink reported as a
# directed glance). Reasoning before concluding is what makes the verdict auditable.
ANALYSIS_PROMPT = """
You are analyzing a video of a person making a recorded consent statement.
Watch and listen to the ENTIRE video, not isolated frames.

Work in three passes, in this order. Do not skip ahead to conclusions.

PASS 1 - OBSERVE. Before judging anything, list what you literally see and hear,
as neutral timestamped facts. Describe gaze direction, eye movements, pauses,
lip movement, audible voices, and anything held or read from - WITHOUT labelling
any of it as prompting or coaching yet. Use plain physical description ("eyes
directed downward toward lap for ~3 seconds", "both eyelids close and reopen in
under 0.3s", "a second voice says 'and conditions'"). This is the evidence base
for everything that follows.

PASS 2 - INTERPRET. For each observation, consider the innocent explanation and
the concerning explanation side by side, and note which the evidence actually
supports. A person recalling memorised words often looks down or away; that is
normal and is not coaching.

PASS 3 - CONCLUDE. Only now assign verdicts, and only from observations you
recorded in Pass 1. Every non-"NO" verdict must cite the observation ids it rests on.

Return ONLY valid JSON (no markdown, no commentary) matching exactly this schema:

{
  "observations": [
    {"id": "o1", "timestamp": "MM:SS-MM:SS", "type": "gaze" | "eye_movement" | "pause" | "voice" | "object" | "other",
     "description": "neutral physical description only, no interpretation",
     "innocent_explanation": "the most plausible benign reading of this observation",
     "concerning_explanation": "the prompting/coaching reading, or null if none is plausible",
     "which_is_better_supported": "innocent" | "concerning" | "cannot_tell"}
  ],
  "prompting_reasoning": "string - weigh the observations against each other and explain how you reached the prompting verdict, referencing observation ids",
  "prompting_detected": "YES" | "NO" | "INDETERMINATE",
  "prompting_evidence": [{"timestamp": "MM:SS-MM:SS", "observation": "string, direct observation only"}],
  "other_person_reasoning": "string - explain the other-person verdict, referencing observation ids",
  "other_person_present": "YES" | "NO" | "INDETERMINATE",
  "other_person_visible": true | false | null,
  "other_person_audible": true | false | null,
  "approx_other_people_count": integer | null,
  "other_person_evidence": [{"timestamp": "MM:SS-MM:SS", "observation": "string"}],
  "environment_description": "concise factual description of visible surroundings",
  "environment_indoor_outdoor": "indoor" | "outdoor" | "indeterminate",
  "socioeconomic_assessment": "concise, evidence-based description of the physical environment only (property condition, furniture, visible goods) - do NOT infer anything about the person's identity, ethnicity, or protected characteristics",
  "socioeconomic_evidence_sufficient": true | false,
  "video_quality": "SUFFICIENT" | "MARGINAL" | "INSUFFICIENT",
  "video_quality_notes": "string - lighting, face visibility, cuts, camera movement, occlusions",
  "continuous_recording": true | false | "INDETERMINATE",
  "overall_visual_audio_confidence": 0.0-1.0
}

Rules:
- Distinguish observation from inference. Only mark something YES if there is direct
  observable evidence. If evidence is ambiguous or absent, use "INDETERMINATE" / null.
- Do NOT assume prompting or another person's presence just because it can't be ruled out.
- Do NOT infer race, ethnicity, religion, or other protected characteristics from appearance.
- Base socioeconomic_assessment strictly on the physical environment, not the person.

Prompting detection - be conservative, this is a high-stakes false-positive risk:
- A blink is NOT evidence of prompting. Distinguish a blink (eyes briefly closing, both
  eyes, symmetric, typically under half a second) from a directed glance (eyes moving
  toward and briefly fixating on a consistent off-camera point, then returning).
- A single brief glance away from the camera, on its own, is normal human behavior
  during unscripted or semi-scripted speech and is NOT sufficient evidence of prompting.
  Do not return "YES" based on one isolated glance.
- Sustained downward gaze is genuinely ambiguous on its own: it may mean reading from
  notes or a phone, or it may just be someone concentrating on recalling memorised
  words, or looking at the device they are recording on. Treat it as INDETERMINATE
  unless corroborated (e.g. visible text/screen being read, eye movement tracking
  line-by-line across something, or speech pausing exactly where a reader would pause).
  Never call it "NO" silently - record the observation and explain the uncertainty.
- Only return "YES" if there is a genuine PATTERN: e.g. multiple glances toward the same
  off-camera direction over the course of the video, OR a glance clearly co-occurring
  with a pause AND a change in what is being said (as if reading or receiving an
  answer), OR a glance clearly correlated with an audible second voice or visible cue.
- A single ambiguous observation (one glance, one pause, one blink) should result in
  "INDETERMINATE", not "YES". Reserve "YES" for cases with multiple corroborating
  observations or unambiguous direct evidence (e.g. another person's voice audible
  immediately feeding the subject words, visible instruction from someone else, subject
  reading from something clearly out of frame).
- Before concluding prompting, ask: could this observation equally be explained by
  normal unscripted speech (blinking, brief downward glance while recalling words,
  natural pauses)? If yes, do not treat it as prompting evidence.

Evidence must justify the verdict - an unexplained verdict is a failure:
- prompting_evidence MUST be non-empty whenever prompting_detected is "YES" OR
  "INDETERMINATE". An INDETERMINATE with an empty evidence array is invalid output: a
  human reviewer has to be able to see what created the doubt. For INDETERMINATE,
  state the observation AND why it is inconclusive, e.g. "subject's gaze is directed
  downward for the majority of the recording rather than at the camera - consistent
  with reading from notes, but equally consistent with recalling memorised words; no
  visible text or screen and no line-tracking eye movement, so cannot confirm either
  way". Leave it empty ONLY when the verdict is "NO" with no notable observations.
- Apply the same standard to other_person_evidence: it must be non-empty whenever
  other_person_present is "YES" or "INDETERMINATE", explaining the basis for the
  uncertainty in the INDETERMINATE case.
- Do not list a blink or a single momentary glance as evidence FOR prompting. If a
  blink or brief glance is the only thing you observed and it does not support a
  prompting verdict, the verdict is "NO" - but still record the observation in
  `observations` so the reasoning is visible.
- prompting_reasoning and other_person_reasoning must never be empty. They must
  reference specific observation ids and explain the weighing, not restate the verdict.
"""


MAX_UPLOAD_POLL_ITERATIONS = 60  # 60 * 2s = 120s cap on Gemini processing wait


# upload video to Gemini and poll until it's processed (Gemini requires this for video)
def upload_and_wait(video_path: str):
    uploaded = client.files.upload(file=video_path)
    iterations = 0
    while uploaded.state.name == "PROCESSING":
        if iterations >= MAX_UPLOAD_POLL_ITERATIONS:
            raise RuntimeError("Gemini file processing timed out")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
        iterations += 1
    if uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini file processing failed for uploaded video")
    return uploaded


# The prompt requires evidence for any non-"NO" verdict, but a model can still return a
# hedged verdict with an empty evidence array. Rather than let that reach a human reviewer
# as an unexplained INDETERMINATE, backfill evidence from the chain-of-thought
# `observations` the model was required to record first - those are the reasoning it
# actually did, so they're the honest explanation for the doubt.
def backfill_missing_evidence(result: dict, verdict_key: str, evidence_key: str, reasoning_key: str) -> None:
    if result.get(verdict_key) not in ("YES", "INDETERMINATE"):
        return
    if result.get(evidence_key):
        return

    # prefer observations the model itself couldn't resolve - those are the actual doubt
    observations = result.get("observations") or []
    unresolved = [o for o in observations if o.get("which_is_better_supported") in ("cannot_tell", "concerning")]
    source = unresolved or observations

    if source:
        result[evidence_key] = [
            {
                "timestamp": o.get("timestamp", "unknown"),
                "observation": o.get("description", "") + (
                    f" (inconclusive: {o['concerning_explanation']})"
                    if o.get("concerning_explanation") else ""
                ),
            }
            for o in source
        ]
    elif result.get(reasoning_key):
        # no structured observations either - fall back to the free-text reasoning so the
        # reviewer at least sees why, rather than a bare verdict
        result[evidence_key] = [{"timestamp": "unknown", "observation": result[reasoning_key]}]


# run the multimodal analysis and parse the structured JSON result
def analyze_video(video_path: str) -> dict:
    uploaded_file = upload_and_wait(video_path)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded_file, ANALYSIS_PROMPT],
        config={"response_mime_type": "application/json"},
    )

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        raise RuntimeError(f"Gemini returned non-JSON response: {e} | raw: {response}")

    backfill_missing_evidence(result, "prompting_detected", "prompting_evidence", "prompting_reasoning")
    backfill_missing_evidence(result, "other_person_present", "other_person_evidence", "other_person_reasoning")
    return result
