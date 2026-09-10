"""
Evaluation harness for the consent video pipeline.

Loads eval/cases.json, and for each case:
  - if the referenced video file does not exist on disk, records
    "NOT RUN - video not found" for every graded dimension (no crash).
  - if it exists, runs the real pipeline functions directly (not via HTTP) and
    compares actual vs expected per graded dimension.

Prints a results table to stdout and writes eval/results.md.

Run from repo root: python eval/run_eval.py
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.stt import transcribe_audio
from app.video_analysis import analyze_video
from app.statement_matcher import evaluate_statement
from app.decision import build_final_decision, build_json_result, fuse_other_person_signal
from app.media_utils import extract_audio

DIMENSIONS = [
    "speech_content",
    "name_verification",
    "language_identification",
    "completeness",
    "word_ordering",
    "prompting_detection",
    "other_person_detection",
    "environment_analysis",
    "overall_decision",
]

DIMENSION_LABELS = {
    "speech_content": "Speech/content verification",
    "name_verification": "Name verification",
    "language_identification": "Language identification",
    "completeness": "Completeness",
    "word_ordering": "Word ordering",
    "prompting_detection": "Prompting detection",
    "other_person_detection": "Other-person detection",
    "environment_analysis": "Environment analysis",
    "overall_decision": "Overall decision",
}

NOT_RUN = "NOT RUN - video not found"


def load_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# run the real pipeline (STT -> statement matcher -> video analysis -> decision)
# directly against a video file on disk, no HTTP involved
def run_pipeline(video_path: str, expected_name: str) -> dict:
    audio_path = extract_audio(video_path)
    try:
        stt_result = transcribe_audio(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    statement_result = evaluate_statement(
        stt_result["transcript"], expected_name, stt_result["language"], stt_result["words"]
    )
    video_result = analyze_video(video_path)
    other_person_fusion = fuse_other_person_signal(stt_result, video_result)
    final = build_final_decision(statement_result, video_result, other_person_fusion)
    json_result = build_json_result(
        expected_name, stt_result, statement_result, video_result, final, other_person_fusion
    )
    return json_result


# compare actual pipeline output against a case's expected fields, dimension by dimension
def grade_case(actual: dict, case: dict) -> dict:
    graded = {}

    graded["speech_content"] = actual["statement_status"] != "UNCLEAR"
    graded["name_verification"] = actual["name_matches"] == (case["expected_statement_status"] != "UNCLEAR")
    graded["language_identification"] = (
        (case["expected_language"] == "mixed" and actual.get("mixed_language_detected"))
        or (case["expected_language"] != "mixed" and actual["language"] and actual["language"].lower().startswith(case["expected_language"][:2]))
    )
    graded["completeness"] = actual["statement_complete"] == (case["expected_statement_status"] in ("CORRECT", "INCORRECT"))
    graded["word_ordering"] = actual["statement_order_correct"] == case["expected_order_correct"]
    graded["prompting_detection"] = (actual["prompting_detected"] == "YES") == case["expected_prompting"]
    graded["other_person_detection"] = (actual["other_person_present"] == "YES") == case["expected_other_person"]
    graded["environment_analysis"] = actual["video_quality"] == case["expected_video_quality"]

    overall_ok = actual["statement_status"] == case["expected_statement_status"]
    if "expected_decision" in case:
        overall_ok = overall_ok and actual["decision"] == case["expected_decision"]
    if "expected_confidence_min" in case:
        overall_ok = overall_ok and actual["confidence"] >= case["expected_confidence_min"]
    graded["overall_decision"] = overall_ok

    return graded


def run_all(cases_path: str, results_md_path: str):
    cases = load_cases(cases_path)
    rows = []

    for case in cases:
        video_path = case["video_path"]
        if not os.path.exists(video_path):
            rows.append({"id": case["id"], "description": case["description"], "graded": {d: NOT_RUN for d in DIMENSIONS}})
            continue

        try:
            # a case with no expected name (e.g. an unrelated-statement case) still
            # needs some name to check against - any name works since it won't match
            expected_name = case["expected_name"] or "Rohit Sharma"
            actual = run_pipeline(video_path, expected_name)
            graded = grade_case(actual, case)
            rows.append({"id": case["id"], "description": case["description"], "graded": graded})
        except Exception as e:
            # one failing case must never kill the whole eval run
            err = f"ERROR: {e}"
            traceback.print_exc()
            rows.append({"id": case["id"], "description": case["description"], "graded": {d: err for d in DIMENSIONS}})

    print_table(rows)
    write_results_md(rows, results_md_path)


def symbol(value) -> str:
    if value == NOT_RUN:
        return "⏳"  # hourglass, not run
    if isinstance(value, str) and value.startswith("ERROR"):
        return "❌"
    return "✅" if value else "❌"


def print_table(rows: list):
    header = ["case"] + DIMENSIONS
    print(" | ".join(header))
    for row in rows:
        line = [row["id"]] + [symbol(row["graded"][d]) for d in DIMENSIONS]
        print(" | ".join(line))
    print()
    print_summary(rows)


def print_summary(rows: list):
    total = len(rows)
    for d in DIMENSIONS:
        passed = sum(1 for r in rows if r["graded"][d] is True)
        print(f"{DIMENSION_LABELS[d]}: {passed}/{total}")


def write_results_md(rows: list, path: str):
    lines = ["# Evaluation results", "", "Regenerate with `python eval/run_eval.py` once real videos are added under `eval/videos/`.", ""]

    header = "| Case | " + " | ".join(DIMENSION_LABELS[d] for d in DIMENSIONS) + " |"
    sep = "|" + "---|" * (len(DIMENSIONS) + 1)
    lines.append(header)
    lines.append(sep)

    for row in rows:
        cells = [row["id"]] + [symbol(row["graded"][d]) for d in DIMENSIONS]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-dimension accuracy")
    lines.append("")
    total = len(rows)
    for d in DIMENSIONS:
        passed = sum(1 for r in rows if r["graded"][d] is True)
        lines.append(f"- {DIMENSION_LABELS[d]}: {passed}/{total}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_all(os.path.join(base_dir, "cases.json"), os.path.join(base_dir, "results.md"))
