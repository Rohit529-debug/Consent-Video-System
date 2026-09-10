# Minimal Devanagari -> Latin transliteration, used ONLY to compare a name written in
# one script against the same name spoken in the other. The caller may supply
# expected_name as "priyesh" while Scribe transcribes the Hindi speech as "प्रियेश";
# without this, cross-script comparison scores ~0 and a correct name reads as a mismatch.
#
# This is deliberately not a full ISO-15919/IAST transliterator. It targets the one job
# it has - producing a rough Latin skeleton close enough for fuzzy name comparison - so
# it favours the spellings people actually use for Indian names (e.g. "sh" not "ś",
# inherent "a" after consonants) over scholarly accuracy.

# independent vowels
VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}

# dependent vowel signs (matras) - these are the combining marks that a naive
# punctuation strip destroys, which is why normalize() must preserve category Mn
MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}

CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l",
    # common nukta forms
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

VIRAMA = "्"          # suppresses the inherent "a" of the preceding consonant
ANUSVARA_CHANDRA = {   # nasalisation marks -> "n" is the usual romanisation
    "ं": "n", "ँ": "n", "ॐ": "om",
}
VISARGA = "ः"


# transliterate Devanagari text to a rough Latin form; non-Devanagari characters
# (already-Latin text, digits, spaces) pass through unchanged
def devanagari_to_latin(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        # consonant + optional nukta already handled via CONSONANTS keys of length 2
        two = ch + nxt
        if two in CONSONANTS:
            out.append(CONSONANTS[two])
            i += 2
            # inherent "a" unless followed by a matra/virama
            after = text[i] if i < len(text) else ""
            if after not in MATRAS and after != VIRAMA:
                out.append("a")
            continue

        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
            i += 1
            after = text[i] if i < len(text) else ""
            # the inherent vowel is dropped before a matra or an explicit virama
            if after not in MATRAS and after != VIRAMA:
                out.append("a")
            continue

        if ch in MATRAS:
            out.append(MATRAS[ch])
            i += 1
            continue

        if ch == VIRAMA:
            i += 1  # already suppressed the inherent vowel above
            continue

        if ch in VOWELS:
            out.append(VOWELS[ch])
            i += 1
            continue

        if ch in ANUSVARA_CHANDRA:
            out.append(ANUSVARA_CHANDRA[ch])
            i += 1
            continue

        if ch == VISARGA:
            out.append("h")
            i += 1
            continue

        # not Devanagari - pass through (covers Latin input, spaces, digits)
        out.append(ch)
        i += 1

    return "".join(out)


# True if the text contains any Devanagari codepoints, so callers can skip
# transliteration work entirely for pure-Latin input
def has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)
