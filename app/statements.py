# Required statement template, broken into ordered required "chunks".
# The matcher checks these chunks appear, in this order, in the transcript.
# {NAME} is substituted with the expected name before matching.

ENGLISH_TEMPLATE = [
    "I am {NAME}",
    "I agree with the terms and conditions",
    "I am fully aware of the rules and regulations",
]

# Romanized Hindi variant, since STT usually transliterates Hindi speech
# to Devanagari, but callers may also send transliterated text.
HINDI_TEMPLATE = [
    "मेरा नाम {NAME} है",
    "मैं नियम और शर्तों से सहमत हूं",
    "मैं नियमों और विनियमों से पूरी तरह अवगत हूं",
]

TEMPLATES = {
    "english": ENGLISH_TEMPLATE,
    "hindi": HINDI_TEMPLATE,
}

# Additional accepted phrasings per chunk index, beyond the canonical template above.
# A speaker conveying the required content in a different-but-equivalent construction has
# still said what was required - the assignment asks us to verify content, not to demand
# one exact word order. Hindi in particular has several natural orders for stating a name
# ("मेरा नाम X है" / "मैं हूँ X" / "मैं X हूँ"), and Hinglish speakers freely mix a Hindi
# frame with English content words. Without these, a valid statement reads as INCOMPLETE.
#
# Keyed by chunk index so it stays aligned with the ordered templates.
CHUNK_VARIANTS = {
    "english": {
        0: ["my name is {NAME}", "I'm {NAME}", "this is {NAME}"],
        1: [
            "I agree to the terms and conditions",
            "I accept the terms and conditions",
        ],
        2: [
            "I am fully aware of the rules and regulation",
            "I am aware of the rules and regulations",
        ],
    },
    "hindi": {
        0: ["मैं हूँ {NAME}", "मैं हूं {NAME}", "मैं {NAME} हूँ", "मेरा नाम {NAME}"],
        1: [
            "मैं नियम और शर्तों से सहमत हूँ",
            "मैं नियमों और शर्तों से सहमत हूं",
        ],
        2: [
            "मैं नियमों और विनियमों से पूरी तरह अवगत हूँ",
            "मैं नियम और विनियम से पूरी तरह अवगत हूं",
        ],
    },
}


# all accepted phrasings for one chunk in one language: the canonical template first,
# then any equivalent variants
def chunk_phrasings(language: str, index: int) -> list:
    phrasings = [TEMPLATES[language][index]]
    phrasings.extend(CHUNK_VARIANTS.get(language, {}).get(index, []))
    return phrasings
