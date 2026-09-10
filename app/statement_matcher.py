import re
import unicodedata
from rapidfuzz import fuzz
from app.statements import TEMPLATES, chunk_phrasings
from app.transliterate import devanagari_to_latin, has_devanagari

FUZZY_MATCH_THRESHOLD = 78  # 0-100, tuned for tolerance to STT noise


# lowercase + strip punctuation so fuzzy matching isn't thrown off by formatting.
# Combining marks must be preserved explicitly: neither `\w` nor str.isalnum() covers
# them, so a naive [^\w\s] strip silently destroys Devanagari vowel signs and viramas
# (मेरा नाम -> मर नम), mangling Hindi text and breaking name matching. Devanagari matras
# span BOTH mark categories - Mn (non-spacing: ु ू े ै ं ँ ्) and Mc (spacing combining:
# ा ि ी ो ौ) - so both must be kept, plus Me for completeness.
MARK_CATEGORIES = {"Mn", "Mc", "Me"}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        ch for ch in text
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch) in MARK_CATEGORIES
    )
    return re.sub(r"\s+", " ", text).strip()


# Name-introduction patterns for both languages, including the constructions speakers
# actually use rather than only the template's wording. Hindi has several valid orders:
# "मेरा नाम <name> है", "मैं हूँ <name>", "मैं <name> हूँ" - a speaker may use any of them.
# An English-only "i am ..." pattern with an [a-z] class can never match Devanagari,
# which previously made every Hindi name read as undetected.
#
# Names are short, so captures are bounded tightly (a name is not 40 chars of clause).
# "मैं <name> हूँ" is deliberately last and tightly bounded: `मैं` also opens the other
# two required chunks ("मैं नियम और शर्तों से..."), so a greedy version of it happily
# captures a whole clause from the wrong part of the transcript as the "name".
NAME_PATTERNS = [
    r"i am ([^\s][^.,]{1,30}?)(?:\.|,| i | my |$)",
    r"my name is ([^\s][^.,]{1,30}?)(?:\.|,| i | my |$)",
    # the trailing copula (है / हूँ) is grammar, not part of the name, so exclude those
    # tokens from the capture - a greedy \S+ swallows them and corrupts the match score
    r"मेरा नाम ((?!है|हूँ|हूं)\S{1,20}(?:\s+(?!है|हूँ|हूं)\S{1,20})?)",
    # "मैं हूँ <name>" - the name runs to the end of the clause. Devanagari has no
    # sentence-cased letters to anchor on and normalize() has already removed the
    # danda/full stop, so bound the capture to Devanagari-only tokens: that stops it
    # bleeding into a following English clause ("प्रियेश i agree with...").
    r"मैं हूँ ([ऀ-ॿ]{1,20}(?:\s+[ऀ-ॿ]{1,20})?)",
    r"मैं हूं ([ऀ-ॿ]{1,20}(?:\s+[ऀ-ॿ]{1,20})?)",
    r"मैं ([ऀ-ॿ]{1,20}(?:\s+[ऀ-ॿ]{1,20})?)\s*(?:हूँ|हूं)",
]


# Compare a name against the transcript in whichever script gives the better signal.
# The caller may pass expected_name in Latin ("priyesh") while the speaker's Hindi is
# transcribed in Devanagari ("प्रियेश"); comparing those directly scores ~0 because they
# share no codepoints, so we also compare a transliterated Latin form of both sides.
def best_name_score(candidate: str, expected: str, scorer) -> float:
    scores = [scorer(candidate, expected)]
    if has_devanagari(candidate) or has_devanagari(expected):
        scores.append(scorer(devanagari_to_latin(candidate), devanagari_to_latin(expected)))
    return max(scores)


# try to find the expected name in the transcript using a name-introduction pattern in
# either language, falling back to fuzzy search for the name string anywhere in the
# transcript. Matching is script-agnostic (see best_name_score).
def extract_name(transcript: str, expected_name: str) -> dict:
    norm_transcript = normalize(transcript)
    norm_expected = normalize(expected_name)

    # Collect EVERY candidate from every pattern at every position, then keep the
    # best-scoring one. Taking the first pattern that happens to fire is wrong: `मैं`
    # opens the other required chunks too, so the first hit can easily be a clause from
    # the middle of the statement ("नियमों और विनियमों से पूरी तरह अवगत") rather than the
    # name. Scoring candidates against the expected name picks the real one instead.
    best_candidate = None
    best_score = -1.0
    for pattern in NAME_PATTERNS:
        for match in re.finditer(pattern, norm_transcript):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            score = best_name_score(candidate, norm_expected, fuzz.token_sort_ratio)
            if score > best_score:
                best_score = score
                best_candidate = candidate

    # a captured candidate that looks nothing like the expected name is more likely a
    # mis-capture than a genuinely wrong name, so fall back to searching the whole
    # transcript - that way a real mismatch still scores low, but a pattern misfire
    # doesn't invent a bogus name_detected
    fallback_score = best_name_score(norm_transcript, norm_expected, lambda a, b: fuzz.partial_ratio(b, a))

    if best_candidate is not None and best_score >= FUZZY_MATCH_THRESHOLD:
        detected_name = best_candidate
        score = best_score
    elif fallback_score > best_score:
        detected_name = None
        score = fallback_score
    else:
        detected_name = best_candidate
        score = max(best_score, 0.0)

    return {
        "name_detected": detected_name,
        "name_matches": score >= FUZZY_MATCH_THRESHOLD,
        "name_match_score": score,
        "name_complete": bool(detected_name) and len(detected_name.split()) >= len(norm_expected.split()),
    }


# build a per-word character-offset index into the normalized joined transcript,
# so a matched substring's char span can be resolved back to word timestamps
def build_word_offsets(words: list) -> list:
    offsets = []
    cursor = 0
    for w in words:
        norm_w = normalize(w.get("text", ""))
        if not norm_w:
            continue
        offsets.append({
            "start_char": cursor,
            "end_char": cursor + len(norm_w),
            "start": w.get("start"),
            "end": w.get("end"),
        })
        cursor += len(norm_w) + 1  # +1 for the joining space
    return offsets


# resolve a matched char span [start, end) in the normalized-joined-words transcript
# to an approximate (start, end) timestamp, by finding words overlapping that span
def resolve_timestamp(word_offsets: list, start_char: int, end_char: int) -> dict:
    covering = [
        w for w in word_offsets
        if w["end_char"] > start_char and w["start_char"] < end_char
    ]
    if not covering:
        return {"start": None, "end": None}
    return {"start": covering[0]["start"], "end": covering[-1]["end"]}


UNCLEAR_MIN_WORDS = 4  # transcripts shorter than this can't contain a real statement
NEAR_MISS_CONTENT_OVERLAP = 0.34  # at least ~1/3 of a chunk's real content words present

# words carrying no real meaning for overlap purposes - covers the connective/pronoun
# vocabulary actually used in ENGLISH_TEMPLATE/HINDI_TEMPLATE, English and transliterated
# Hindi. Excluding these keeps content_word_overlap from being fooled by short common
# words matching by coincidence (the same failure mode partial_ratio has at the char level).
STOPWORDS = {
    "i", "am", "is", "are", "the", "a", "an", "of", "and", "with", "my", "to",
    "that", "this", "fully", "so",
    "मैं", "है", "हूं", "हूँ", "से", "और", "के", "की", "का", "मेरा", "नाम", "पूरी", "तरह",
    # transliterated forms of the above - content overlap is also evaluated on
    # transliterated text when chunk and transcript scripts differ
    "main", "hai", "hoon", "hun", "se", "aur", "aura", "ke", "kee", "ka",
    "mera", "meraa", "naam", "naama", "pooree", "puri", "taraha", "tarah",
}


# split a chunk into its meaningful "content words" (drop stopwords), keeping the
# expected name's tokens since those are exactly what we need to detect overlap on
def content_words(chunk: str) -> list:
    return [w for w in normalize(chunk).split() if w not in STOPWORDS]


# A fuzzy chunk match must also be backed by real word-level content overlap. partial_ratio
# aligns the best-matching substring, so it can return 100 for a chunk whose words are
# nowhere in the transcript (e.g. the Hindi "terms and conditions" chunk scoring 100 against
# a transcript that states that clause in English). Requiring half a chunk's content words
# to actually appear stops a phantom match from claiming a chunk - and, because variants of
# the same chunk compete on score, stops it from stealing the language attribution too.
MIN_CONTENT_OVERLAP_FOR_MATCH = 0.5


# fraction (0.0-1.0) of a chunk's content words that appear as a WHOLE WORD anywhere
# in the transcript (allowing a small per-word fuzzy tolerance for STT misspellings).
# This is deliberately word-level, not substring/char-level like find_chunk_in_transcript's
# partial_ratio - char-level similarity can score deceptively high on totally unrelated
# text just from coincidental shared substrings, which is not evidence of real content
# overlap.
# Per-word fuzzy tolerance. Deliberately loose enough to survive real STT artifacts -
# disfluency stutters in particular ("विनि-विनियमों" for "विनियमों" scores ~80) - while
# still far above the score an unrelated word would reach. Whole-chunk phantom matching is
# prevented by requiring a FRACTION of content words (MIN_CONTENT_OVERLAP_FOR_MATCH), so
# this per-word bar doesn't need to carry that job on its own.
WORD_MATCH_RATIO = 75


def content_word_overlap(chunk: str, transcript_words: list) -> float:
    words = content_words(chunk)
    if not words:
        return 0.0
    found = 0
    for word in words:
        # substring check catches stutters/compounds where the expected word is embedded
        if any(
            word == tw or word in tw or fuzz.ratio(word, tw) >= WORD_MATCH_RATIO
            for tw in transcript_words
        ):
            found += 1
    return found / len(words)


# find the best match (substring or fuzzy) for a single chunk anywhere in the transcript.
# returns {"score", "start_char", "end_char", "matched_text"} always (score may be 0),
# with start_char/end_char/matched_text set to None when below FUZZY_MATCH_THRESHOLD -
# callers check start_char is not None to know whether the chunk was "found".
def find_chunk_in_transcript(norm_transcript: str, norm_chunk: str) -> dict:
    idx = norm_transcript.find(norm_chunk)
    if idx != -1:
        return {
            "score": 100,
            "start_char": idx,
            "end_char": idx + len(norm_chunk),
            "matched_text": norm_chunk,
        }

    best_score, best_idx, best_window = best_fuzzy_window(norm_transcript, norm_chunk)
    if best_score >= FUZZY_MATCH_THRESHOLD:
        return {
            "score": best_score,
            "start_char": best_idx,
            "end_char": best_idx + len(best_window),
            "matched_text": best_window,
        }
    return {"score": best_score, "start_char": None, "end_char": None, "matched_text": None}


# Match a chunk allowing for script mismatch. The Hindi template's name slot is filled
# with whatever script the caller supplied (often Latin "priyesh"), so a Hindi chunk like
# "मेरा नाम priyesh है" cannot match a fully-Devanagari transcript on raw codepoints. When
# a direct match fails and the two sides differ in script, retry on transliterated Latin
# copies of both.
#
# A transliterated match reports character offsets into the TRANSLITERATED string, which
# are meaningless as offsets into the original. Mixing the two coordinate spaces across
# chunks silently corrupts the word-order comparison (chunk A's offset measured in
# transliterated space vs chunk B's in original space), so every match also carries
# `word_index`: the index of its first word within the original transcript's word list.
# Transliteration is word-preserving and order-preserving, so word_index is directly
# comparable across both paths and is what the order check must use. `approx_offsets`
# still marks matches whose char offsets can't be used for timestamp resolution.
def find_chunk_any_script(norm_transcript: str, norm_chunk: str) -> dict:
    direct = find_chunk_in_transcript(norm_transcript, norm_chunk)
    direct["approx_offsets"] = False
    if direct["start_char"] is not None and _content_backed(norm_chunk, direct["matched_text"]):
        direct["word_index"] = _char_to_word_index(norm_transcript, direct["start_char"])
        return direct

    # Always worth retrying transliterated, not only when the two sides differ wholesale
    # in script: a chunk can be MIXED (the Hindi template with a Latin-filled name slot,
    # "मैं हूँ priyesh"), so has_devanagari is true on both sides while the name itself
    # still needs transliteration to line up with Devanagari speech.
    if has_devanagari(norm_transcript) or has_devanagari(norm_chunk):
        lat_transcript = devanagari_to_latin(norm_transcript)
        lat_chunk = devanagari_to_latin(norm_chunk)
        translit = find_chunk_in_transcript(lat_transcript, lat_chunk)
        translit["approx_offsets"] = True
        if translit["start_char"] is not None and _content_backed(lat_chunk, translit["matched_text"]):
            # word index in the transliterated string == word index in the original
            translit["word_index"] = _char_to_word_index(lat_transcript, translit["start_char"])
            return translit
        if translit["score"] > direct["score"]:
            return _not_found(translit["score"])

    # a char-level match with no real word overlap is a phantom: report the score for
    # UNCLEAR-vs-INCORRECT purposes, but do not count the chunk as found
    return _not_found(direct["score"])


def _not_found(score: float) -> dict:
    return {
        "score": score,
        "start_char": None,
        "end_char": None,
        "matched_text": None,
        "approx_offsets": False,
        "word_index": None,
    }


# index of the word containing char offset `pos`, counting words the same way split() does
def _char_to_word_index(text: str, pos: int) -> int:
    return len(text[:pos].split())


# Does the MATCHED WINDOW actually contain this chunk's content words? Checking against
# the whole transcript is too weak: a short chunk whose only content word is the name
# ("मैं हूँ priyesh") passes on the name appearing anywhere, letting partial_ratio anchor
# the match on an unrelated clause elsewhere - which then reports a bogus position and
# makes a correctly-ordered statement look reordered. Scoping the check to the window the
# match actually landed on ties the verdict to that location.
def _content_backed(norm_chunk: str, matched_text: str) -> bool:
    if not content_words(norm_chunk):
        return True  # nothing substantive to verify (chunk is all stopwords)
    if not matched_text:
        return False
    return content_word_overlap(norm_chunk, matched_text.split()) >= MIN_CONTENT_OVERLAP_FOR_MATCH


# for each required chunk, try matching it against BOTH the english and hindi
# version of that chunk (mixed-language transcripts are common), independently,
# anywhere in the full transcript (no forward-only restriction).
def check_word_order_and_completeness(transcript: str, expected_name: str, language: str = None, words: list = None) -> dict:
    # `language` is kept for call-site compatibility but unused: each chunk is now
    # matched against both english and hindi independently (mixed-language support).
    english_chunks = [c.replace("{NAME}", expected_name) for c in TEMPLATES["english"]]
    num_chunks = len(english_chunks)

    norm_transcript = normalize(transcript)
    word_offsets = build_word_offsets(words) if words else []

    positions = []          # found WORD index per chunk, or None
    language_per_chunk = [] # language matched per chunk, or None
    missing_chunks = []
    matched_chunks = []
    statement_evidence = []
    best_scores = []  # best fuzzy score per chunk regardless of whether it crossed threshold
    content_overlap_scores = []  # word-level content overlap for the best-scoring chunk variant
    transcript_words = norm_transcript.split()

    for i in range(num_chunks):
        # every accepted phrasing of this chunk in both languages - a speaker who used an
        # equivalent construction ("मैं हूँ X" instead of "मेरा नाम X है") still said the
        # required content, so all variants are legitimate ways to satisfy the chunk
        candidates = [
            (lang, phrasing.replace("{NAME}", expected_name))
            for lang in ("english", "hindi")
            for phrasing in chunk_phrasings(lang, i)
        ]
        best = None
        best_lang = None
        best_display_chunk = None
        for lang, chunk_text in candidates:
            norm_chunk = normalize(chunk_text)
            result = find_chunk_any_script(norm_transcript, norm_chunk)
            # Rank candidates by (actually found, score, same-script). "Found" must
            # dominate raw score: a phantom high-scoring variant that failed the content
            # check would otherwise beat a genuine lower-scoring match and take the chunk
            # (and its language attribution) with it. Same-script breaks remaining ties so
            # language_per_chunk reports the language actually spoken.
            rank = (
                result["word_index"] is not None,
                result["score"],
                not result["approx_offsets"],
            )
            best_rank = None if best is None else (
                best["word_index"] is not None,
                best["score"],
                not best["approx_offsets"],
            )
            if best_rank is None or rank > best_rank:
                best = result
                best_lang = lang
                best_display_chunk = chunk_text

        best_scores.append(best["score"])
        content_overlap_scores.append(content_word_overlap(best_display_chunk, transcript_words))

        if best["word_index"] is not None:
            # order is compared by WORD index, which is consistent whether the match came
            # from the original or the transliterated string (see find_chunk_any_script)
            positions.append(best["word_index"])
            language_per_chunk.append(best_lang)
            matched_chunks.append(best_display_chunk)
            # offsets from a cross-script (transliterated) match don't index the original
            # transcript, so timestamps can't be resolved from them
            can_timestamp = word_offsets and not best["approx_offsets"]
            ts = resolve_timestamp(word_offsets, best["start_char"], best["end_char"]) if can_timestamp else {"start": None, "end": None}
            statement_evidence.append({
                "chunk": best_display_chunk,
                "start": ts["start"],
                "end": ts["end"],
                "matched_text": best["matched_text"],
            })
        else:
            positions.append(None)
            language_per_chunk.append(None)
            # english template used as the canonical label for the missing chunk
            missing_chunks.append(english_chunks[i])

    found_indices = [i for i, p in enumerate(positions) if p is not None]
    found_positions = [positions[i] for i in found_indices]

    # order is judged only among chunks that were actually found
    order_correct = found_positions == sorted(found_positions)

    # chunks that were found but sit out of expected sequence relative to other found chunks:
    # a found chunk is "reordered" if its rank by transcript position differs from its
    # rank by expected template order (both restricted to the found subset)
    reordered_chunks = []
    if not order_correct:
        actual_order = sorted(found_indices, key=lambda i: positions[i])
        for rank, i in enumerate(found_indices):
            if actual_order.index(i) != rank:
                reordered_chunks.append(matched_chunks[found_indices.index(i)])

    matched_languages = {lang for lang in language_per_chunk if lang is not None}
    mixed_language_detected = len(matched_languages) > 1

    if not missing_chunks and order_correct:
        status = "CORRECT"
    elif not found_indices:
        # UNCLEAR vs INCORRECT: zero chunks matched, but that's ambiguous on its own -
        # it could mean the audio was unintelligible (UNCLEAR) or it could mean the
        # speaker clearly said something coherent that simply isn't the required
        # statement (INCORRECT). We only call it UNCLEAR when there's a real signal
        # the transcript itself can't be trusted: empty/near-empty, or so short it
        # couldn't contain a statement. Otherwise, a substantive transcript where every
        # chunk's real CONTENT WORDS (not stopwords) are largely absent is evidence of
        # coherent-but-unrelated speech, not unintelligible audio. This is deliberately
        # word-level rather than char-level fuzzy (best_scores/partial_ratio): char-level
        # similarity can score deceptively high on unrelated text from coincidental
        # shared substrings/short words, which previously caused unrelated statements to
        # be misclassified as UNCLEAR instead of INCORRECT.
        word_count = len(transcript_words)
        all_unrelated = all(s < NEAR_MISS_CONTENT_OVERLAP for s in content_overlap_scores) if content_overlap_scores else True
        transcript_is_substantive = word_count >= UNCLEAR_MIN_WORDS
        if transcript_is_substantive and all_unrelated:
            status = "INCORRECT"
        else:
            status = "UNCLEAR"
    elif missing_chunks and order_correct:
        status = "INCOMPLETE"
    else:
        # order wrong: either all found (reordered) or some missing + wrong order
        status = "INCORRECT"

    return {
        "statement_status": status,
        "statement_complete": len(missing_chunks) == 0,
        "statement_order_correct": order_correct,
        "missing_portions": missing_chunks,
        "missing_chunks": missing_chunks,
        "matched_chunks": matched_chunks,
        "reordered_chunks": reordered_chunks,
        "language_per_chunk": language_per_chunk,
        "mixed_language_detected": mixed_language_detected,
        "statement_evidence": statement_evidence,
    }


# slide a window of similar length across the transcript looking for the best fuzzy match
def best_fuzzy_window(transcript: str, chunk: str) -> tuple:
    words = transcript.split()
    chunk_len = max(len(chunk.split()), 1)
    best_score = 0
    best_coverage = -1.0
    best_idx = -1
    best_window = ""
    for i in range(len(words)):
        window = " ".join(words[i:i + chunk_len + 2])
        # partial_ratio alone picks badly here: it saturates at 100 for any window merely
        # CONTAINING a good alignment, so a short window aligned on the chunk's tail
        # ("...से पूरी तरह अवगत हूँ") beats the full clause whenever the full clause carries
        # an STT artifact (a stutter like "विनि-विनियमों") that costs it a few points.
        # Rank primarily by how much of the chunk's actual content the window covers, and
        # use the char-level score only to break ties between equally-covering windows.
        coverage = content_word_overlap(chunk, window.split())
        score = fuzz.partial_ratio(chunk, window)
        if (coverage, score) > (best_coverage, best_score):
            best_score = score
            best_coverage = coverage
            best_idx = transcript.find(window)
            best_window = window
    return best_score, best_idx, best_window


# simple heuristic flags for self-correction / hesitation, useful supporting evidence
def detect_hesitation_and_correction(transcript: str) -> dict:
    lower = transcript.lower()
    hesitation_markers = ["um", "uh", "erm", "i mean", "sorry", "wait"]
    correction_markers = ["i mean", "sorry", "let me start again", "no wait", "actually"]

    hesitation_found = [m for m in hesitation_markers if re.search(rf"\b{re.escape(m)}\b", lower)]
    correction_found = [m for m in correction_markers if re.search(rf"\b{re.escape(m)}\b", lower)]

    return {
        "hesitation_detected": len(hesitation_found) > 0,
        "hesitation_markers": hesitation_found,
        "self_correction_detected": len(correction_found) > 0,
        "correction_markers": correction_found,
    }


# top-level entry point combining name + order/completeness + hesitation checks.
# `words` (ElevenLabs word-level timestamps) is optional so callers/tests can omit it.
def evaluate_statement(transcript: str, expected_name: str, language: str, words: list = None) -> dict:
    name_result = extract_name(transcript, expected_name)
    order_result = check_word_order_and_completeness(transcript, expected_name, language, words)
    hesitation_result = detect_hesitation_and_correction(transcript)

    return {**name_result, **order_result, **hesitation_result}
