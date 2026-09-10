# cross-check audio diarization (distinct speaker count) against Gemini's visual/audio
# read of "other person present". Diarization can hear a second voice Gemini's own
# audio track missed being confident about (or vice versa) - when they disagree in the
# "diarization says >1 speaker but Gemini says NO" direction, that's a real conflict
# worth surfacing rather than silently trusting either side.
def fuse_other_person_signal(stt_result: dict, video_result: dict) -> dict:
    distinct_speakers = stt_result.get("distinct_speakers_detected")
    gemini_value = video_result.get("other_person_present")

    conflict = distinct_speakers is not None and distinct_speakers > 1 and gemini_value == "NO"

    if conflict:
        fused_value = "INDETERMINATE"
        note = (
            f"Diarization detected {distinct_speakers} distinct speakers but Gemini's "
            f"visual/audio analysis reported no other person present - flagged for review."
        )
    elif distinct_speakers is not None and distinct_speakers > 1 and gemini_value in ("YES", "INDETERMINATE"):
        fused_value = "YES"
        note = None
    else:
        fused_value = gemini_value
        note = None

    return {
        "other_person_present_fused": fused_value,
        "distinct_speakers_detected": distinct_speakers,
        "other_person_audio_video_conflict": conflict,
        "other_person_conflict_note": note,
    }


# decide the final PASS / FAIL / REVIEW outcome from all component results
def build_final_decision(statement_result: dict, video_result: dict, other_person_fusion: dict) -> dict:
    fused_other_person = other_person_fusion["other_person_present_fused"]

    hard_fail_conditions = [
        statement_result["statement_status"] in ("INCORRECT", "UNCLEAR"),
        statement_result["name_matches"] is False,
        video_result["prompting_detected"] == "YES",
        fused_other_person == "YES",
        video_result["video_quality"] == "INSUFFICIENT",
    ]

    needs_review_conditions = [
        statement_result["statement_status"] == "INCOMPLETE",
        video_result["prompting_detected"] == "INDETERMINATE",
        fused_other_person == "INDETERMINATE",
        video_result["video_quality"] == "MARGINAL",
        statement_result["hesitation_detected"],
        other_person_fusion["other_person_audio_video_conflict"],
    ]

    if any(hard_fail_conditions):
        decision = "FAIL"
    elif any(needs_review_conditions):
        decision = "REVIEW"
    else:
        decision = "PASS"

    confidence = compute_overall_confidence(statement_result, video_result)

    return {
        "decision": decision,
        "confidence": confidence,
    }


# Confidence reflects how sure the system is about the FINAL decision, not an average
# of how many sub-checks came back green. A clean FAIL (e.g. a coherent transcript that
# clearly doesn't match the required statement) should be just as high-confidence as a
# clean PASS - it's the same kind of decisive, unambiguous judgment. Averaging that
# against unrelated positive signals (e.g. good lighting) would wrongly dilute a
# confident wrong answer toward a mediocre score.
#
# So we split the score into two parts:
#   1. primary_confidence - how decisive the core judgment is: the statement match
#      (CORRECT/INCORRECT are both decisive; INCOMPLETE/UNCLEAR are inherently
#      ambiguous) plus Gemini's prompting/other-person reads (a firm YES or NO is
#      decisive, INDETERMINATE is not).
#   2. quality_discount - only applied when there's a genuine quality/ambiguity flag:
#      MARGINAL/INSUFFICIENT video, any INDETERMINATE sub-result, or a fuzzy name/chunk
#      match score sitting in the borderline zone (neither clearly matching nor clearly
#      not). A SUFFICIENT-quality video with a decisive right-or-wrong statement is not
#      discounted just because it also has other sub-scores.
def compute_overall_confidence(statement_result: dict, video_result: dict) -> float:
    status = statement_result.get("statement_status")
    name_score = statement_result.get("name_match_score", 0)

    # decisiveness of the statement judgment: CORRECT/INCORRECT are firm outcomes,
    # INCOMPLETE/UNCLEAR are inherently hedged and should not score as "confident"
    if status in ("CORRECT", "INCORRECT"):
        statement_confidence = 1.0
    elif status == "INCOMPLETE":
        statement_confidence = 0.6
    else:  # UNCLEAR
        statement_confidence = 0.4

    prompting = video_result.get("prompting_detected")
    other_person = video_result.get("other_person_present")
    prompting_confidence = 0.5 if prompting == "INDETERMINATE" else 1.0
    other_person_confidence = 0.5 if other_person == "INDETERMINATE" else 1.0

    primary_confidence = (
        0.6 * statement_confidence + 0.2 * prompting_confidence + 0.2 * other_person_confidence
    )

    # borderline name/fuzzy match score: not clearly matching, not clearly failing -
    # genuine ambiguity in the evidence itself, so it earns a discount
    name_is_borderline = 60 <= name_score < 90

    quality_flags = [
        video_result.get("video_quality") in ("MARGINAL", "INSUFFICIENT"),
        prompting == "INDETERMINATE",
        other_person == "INDETERMINATE",
        name_is_borderline,
    ]
    quality_discount = 0.15 * sum(quality_flags)

    confidence = max(0.0, min(1.0, primary_confidence - quality_discount))
    return round(confidence, 2)


# assemble the full machine-readable JSON result matching the assignment's example schema
def build_json_result(expected_name, stt_result, statement_result, video_result, final, other_person_fusion) -> dict:
    return {
        "statement_status": statement_result["statement_status"],
        "language": stt_result["language"],
        "language_per_chunk": statement_result["language_per_chunk"],
        "mixed_language_detected": statement_result["mixed_language_detected"],
        "name_detected": statement_result["name_detected"],
        "name_matches": statement_result["name_matches"],
        "statement_complete": statement_result["statement_complete"],
        "statement_order_correct": statement_result["statement_order_correct"],
        "missing_portions": statement_result["missing_portions"],
        "reordered_chunks": statement_result["reordered_chunks"],
        "statement_evidence": statement_result["statement_evidence"],
        "hesitation_detected": statement_result["hesitation_detected"],
        "self_correction_detected": statement_result["self_correction_detected"],
        "prompting_detected": video_result["prompting_detected"],
        "prompting_evidence": video_result["prompting_evidence"],
        "prompting_reasoning": video_result.get("prompting_reasoning"),
        "other_person_present": other_person_fusion["other_person_present_fused"],
        "other_person_visible": video_result["other_person_visible"],
        "other_person_audible": video_result["other_person_audible"],
        "approx_other_people_count": video_result["approx_other_people_count"],
        "other_person_evidence": video_result["other_person_evidence"],
        "other_person_reasoning": video_result.get("other_person_reasoning"),
        "visual_observations": video_result.get("observations", []),
        "distinct_speakers_detected": other_person_fusion["distinct_speakers_detected"],
        "other_person_audio_video_conflict": other_person_fusion["other_person_audio_video_conflict"],
        "video_quality": video_result["video_quality"],
        "video_quality_notes": video_result["video_quality_notes"],
        "continuous_recording": video_result["continuous_recording"],
        "environment": {
            "description": video_result["environment_description"],
            "indoor_outdoor": video_result["environment_indoor_outdoor"],
            "socioeconomic_assessment": video_result["socioeconomic_assessment"],
            "evidence_sufficient": video_result["socioeconomic_evidence_sufficient"],
        },
        "transcript": stt_result["transcript"],
        "decision": final["decision"],
        "confidence": final["confidence"],
        "expected_name": expected_name,
    }


# build the concise human-readable report described in the assignment
def build_human_report(result: dict) -> str:
    lines = [
        f"DECISION: {result['decision']}  (confidence: {result['confidence']})",
        "",
        f"Transcript: \"{result['transcript']}\"",
        f"Detected language: {result['language']}",
        "",
        f"Name: {result['name_detected']} (expected: {result['expected_name']}) "
        f"-> {'MATCH' if result['name_matches'] else 'NO MATCH'}",
        f"Statement status: {result['statement_status']} "
        f"(complete: {result['statement_complete']}, order correct: {result['statement_order_correct']})",
    ]
    if result["missing_portions"]:
        lines.append(f"Missing portions: {result['missing_portions']}")
    if result["reordered_chunks"]:
        lines.append(f"Reordered chunks (present but out of sequence): {result['reordered_chunks']}")
    if result["mixed_language_detected"]:
        lines.append(f"Mixed language detected across statement chunks: {result['language_per_chunk']}")
    if result["hesitation_detected"] or result["self_correction_detected"]:
        lines.append("Note: hesitation or self-correction detected in speech.")
    if result["statement_evidence"]:
        lines.append("Statement evidence (timestamped):")
        for ev in result["statement_evidence"]:
            lines.append(f"  - [{ev['start']}-{ev['end']}] \"{ev['chunk']}\" -> matched: \"{ev['matched_text']}\"")

    lines += [
        "",
        f"Prompting detected: {result['prompting_detected']}",
    ]
    if result.get("prompting_reasoning"):
        lines.append(f"  Reasoning: {result['prompting_reasoning']}")
    if result.get("prompting_evidence"):
        lines.append("  Prompting evidence:")
        for ev in result["prompting_evidence"]:
            lines.append(f"    - [{ev.get('timestamp', 'unknown')}] {ev.get('observation', '')}")

    lines.append(
        f"Other person present: {result['other_person_present']} "
        f"(visible: {result['other_person_visible']}, audible: {result['other_person_audible']}, "
        f"distinct speakers detected: {result['distinct_speakers_detected']})"
    )
    if result.get("other_person_reasoning"):
        lines.append(f"  Reasoning: {result['other_person_reasoning']}")
    if result.get("other_person_evidence"):
        lines.append("  Other person evidence:")
        for ev in result["other_person_evidence"]:
            lines.append(f"    - [{ev.get('timestamp', 'unknown')}] {ev.get('observation', '')}")
    if result["other_person_audio_video_conflict"]:
        lines.append("Note: audio diarization and Gemini's visual/audio read of other-person presence conflict - flagged for review.")
    lines += [
        "",
        f"Environment: {result['environment']['description']}",
        f"Socioeconomic assessment: {result['environment']['socioeconomic_assessment']}",
        f"(evidence sufficient: {result['environment']['evidence_sufficient']})",
        "",
        f"Video quality: {result['video_quality']} - {result['video_quality_notes']}",
        f"Continuous recording: {result['continuous_recording']}",
    ]
    return "\n".join(lines)
