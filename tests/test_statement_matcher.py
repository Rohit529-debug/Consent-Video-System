import pytest
from app.statement_matcher import evaluate_statement, extract_name, normalize
from app.statements import TEMPLATES
from app.transliterate import devanagari_to_latin

# a real Devanagari consent statement as ElevenLabs Scribe actually transcribes it
HINDI_SPOKEN = (
    "मेरा नाम प्रियेश है। मैं नियम और शर्तों से सहमत हूँ "
    "और मैं नियमों और विनियमों से पूरी तरह अवगत हूँ।"
)

NAME = "Rohit Sharma"

ENGLISH_CHUNKS = [c.replace("{NAME}", NAME) for c in TEMPLATES["english"]]
HINDI_CHUNKS = [c.replace("{NAME}", NAME) for c in TEMPLATES["hindi"]]


def test_correct_english_statement():
    transcript = ". ".join(ENGLISH_CHUNKS) + "."
    result = evaluate_statement(transcript, NAME, "english")
    assert result["statement_status"] == "CORRECT"
    assert result["name_matches"] is True
    assert result["statement_complete"] is True
    assert result["statement_order_correct"] is True
    assert result["mixed_language_detected"] is False


def test_correct_hindi_statement():
    transcript = ". ".join(HINDI_CHUNKS) + "."
    result = evaluate_statement(transcript, NAME, "hindi")
    assert result["statement_status"] == "CORRECT"
    assert result["statement_complete"] is True
    assert result["statement_order_correct"] is True


def test_mixed_language_statement():
    # first chunk in english, second in hindi, third in english
    transcript = f"{ENGLISH_CHUNKS[0]}. {HINDI_CHUNKS[1]}. {ENGLISH_CHUNKS[2]}."
    result = evaluate_statement(transcript, NAME, "auto")
    assert result["statement_status"] == "CORRECT"
    assert result["mixed_language_detected"] is True
    assert result["language_per_chunk"] == ["english", "hindi", "english"]


def test_missing_chunk_is_incomplete():
    # drop the "terms and conditions" chunk entirely
    transcript = f"{ENGLISH_CHUNKS[0]}. {ENGLISH_CHUNKS[2]}."
    result = evaluate_statement(transcript, NAME, "english")
    assert result["statement_status"] == "INCOMPLETE"
    assert ENGLISH_CHUNKS[1] in result["missing_portions"]
    assert ENGLISH_CHUNKS[1] in result["missing_chunks"]


def test_reordered_chunks_all_present():
    # all three chunks present but in swapped order (2nd and 1st swapped)
    transcript = f"{ENGLISH_CHUNKS[1]}. {ENGLISH_CHUNKS[0]}. {ENGLISH_CHUNKS[2]}."
    result = evaluate_statement(transcript, NAME, "english")
    assert result["statement_status"] == "INCORRECT"
    assert result["statement_complete"] is True
    assert len(result["reordered_chunks"]) > 0
    assert result["missing_portions"] == []


def test_wrong_name_does_not_match():
    transcript = ". ".join(ENGLISH_CHUNKS) + "."
    result = extract_name(transcript, "Someone Else")
    assert result["name_matches"] is False


def test_partial_name_not_complete():
    transcript = "I am Rohit. I agree with the terms and conditions."
    result = extract_name(transcript, NAME)
    assert result["name_complete"] is False


def test_empty_transcript_is_unclear():
    result = evaluate_statement("", NAME, "english")
    assert result["statement_status"] == "UNCLEAR"


def test_near_empty_transcript_is_unclear():
    result = evaluate_statement("uh um", NAME, "english")
    assert result["statement_status"] == "UNCLEAR"


def test_coherent_unrelated_statement_is_incorrect():
    # clean, substantive, clearly-transcribed speech that has nothing to do with
    # the required consent statement - should be a confident INCORRECT, not UNCLEAR
    transcript = (
        "Congratulations on twenty five years of partnership between our two "
        "companies, we look forward to many more years of collaboration together."
    )
    result = evaluate_statement(transcript, NAME, "english")
    assert result["statement_status"] == "INCORRECT"
    assert result["statement_complete"] is False


def test_unrelated_statement_regression_case_is_incorrect():
    # regression case: char-level partial_ratio previously scored this unrelated
    # transcript as a "near miss" against a chunk purely from coincidental shared
    # substrings, wrongly producing UNCLEAR instead of INCORRECT
    transcript = "Congratulations for completing 25 years of strong partnership with Aditya Birla Money. Cheers!"
    result = evaluate_statement(transcript, "Priyesh", "english")
    assert result["statement_status"] == "INCORRECT"


def test_garbled_attempt_at_real_statement():
    # a genuinely garbled (misspelled) attempt at the actual statement should still
    # be recognized via the existing fuzzy chunk matcher, not sent down the
    # UNCLEAR/INCORRECT content-overlap path at all
    transcript = "I am Preyash. I gree with term end condition and I fully avare of rule and regulation."
    result = evaluate_statement(transcript, "Priyesh", "english")
    assert result["statement_status"] in ("CORRECT", "INCOMPLETE")


# --- Devanagari / cross-script handling ---

def test_normalize_preserves_devanagari_vowel_marks():
    # regression: [^\w\s] stripping destroyed matras (मेरा नाम -> मर नम), because
    # Devanagari vowel signs span BOTH Mn and Mc unicode mark categories
    result = normalize("मेरा नाम प्रियेश है।")
    assert "मेरा" in result
    assert "नाम" in result
    assert "प्रियेश" in result


def test_transliteration_roundtrip_for_name():
    assert devanagari_to_latin("प्रियेश").startswith("priyesh")


def test_hindi_statement_with_latin_expected_name():
    # the reported failure: caller supplies the name in Latin, speaker says it in
    # Devanagari - must still match rather than reporting a name mismatch
    result = evaluate_statement(HINDI_SPOKEN, "priyesh", "hindi")
    assert result["statement_status"] == "CORRECT"
    assert result["name_matches"] is True
    assert result["name_detected"] is not None


def test_hindi_statement_with_devanagari_expected_name():
    result = evaluate_statement(HINDI_SPOKEN, "प्रियेश", "hindi")
    assert result["statement_status"] == "CORRECT"
    assert result["name_matches"] is True


def test_hindi_statement_with_wrong_name_still_fails():
    # the cross-script fix must not make name checking permissive
    result = evaluate_statement(HINDI_SPOKEN, "rajesh kumar", "hindi")
    assert result["name_matches"] is False


def test_hindi_name_extracted_from_mera_naam_pattern():
    result = extract_name("मेरा नाम प्रियेश है।", "priyesh")
    assert result["name_detected"] is not None
    assert result["name_matches"] is True


def test_hindi_name_capture_excludes_trailing_copula():
    # "है" is grammar, not part of the name - a greedy capture swallows it and
    # depresses the match score
    result = extract_name("मेरा नाम प्रियेश है।", "priyesh")
    assert "है" not in (result["name_detected"] or "")


# --- Hinglish / code-switching ---

# the reported failure: Hindi name frame, English middle chunk, Hindi final chunk,
# with an STT disfluency stutter ("विनि-विनियमों") in the last chunk
HINGLISH_SPOKEN = (
    "मैं हूँ प्रियेश। I agree with the terms and condition "
    "और मैं नियमों और विनि-विनियमों से पूरी तरह अवगत हूँ।"
)


def test_hinglish_statement_is_correct_and_not_reordered():
    result = evaluate_statement(HINGLISH_SPOKEN, "priyesh", "hindi")
    assert result["statement_status"] == "CORRECT"
    assert result["name_matches"] is True
    assert result["statement_complete"] is True
    # regression: chunk positions were compared across two different coordinate spaces
    # (original vs transliterated), making a correctly-ordered statement look reordered
    assert result["statement_order_correct"] is True
    assert result["reordered_chunks"] == []


def test_hinglish_language_attributed_per_chunk():
    result = evaluate_statement(HINGLISH_SPOKEN, "priyesh", "hindi")
    assert result["language_per_chunk"] == ["hindi", "english", "hindi"]
    assert result["mixed_language_detected"] is True


def test_alternate_hindi_name_construction_accepted():
    # "मैं हूँ <name>" conveys the required content even though the template says
    # "मेरा नाम <name> है"
    transcript = (
        "मैं हूँ प्रियेश। मैं नियम और शर्तों से सहमत हूँ "
        "और मैं नियमों और विनियमों से पूरी तरह अवगत हूँ।"
    )
    result = evaluate_statement(transcript, "priyesh", "hindi")
    assert result["statement_status"] == "CORRECT"


def test_phantom_cross_language_chunk_not_counted():
    # the Hindi "terms and conditions" chunk must NOT match a transcript that states
    # that clause only in English - partial_ratio scores it 100 on char overlap alone
    transcript = "मैं हूँ प्रियेश। I agree with the terms and conditions."
    result = evaluate_statement(transcript, "priyesh", "hindi")
    assert result["statement_status"] == "INCOMPLETE"
    # the awareness chunk is genuinely absent, and must be reported as missing
    assert any("aware" in c or "अवगत" in c for c in result["missing_portions"])


def test_stutter_disfluency_still_matches_chunk():
    # STT disfluency artifacts must not break chunk matching
    transcript = "मैं नियमों और विनि-विनियमों से पूरी तरह अवगत हूँ।"
    result = evaluate_statement(transcript, "priyesh", "hindi")
    assert "मैं नियमों और विनियमों से पूरी तरह अवगत हूं" not in result["missing_portions"]
