"""Regex pattern engine for fast message classification.

Matches common bot responses to audio library keys without needing an LLM call.
Language-agnostic: Turkish + English patterns built-in.
"""

import re
from typing import Optional

# Compiled pattern cache (built on first import)
_compiled: dict[str, list[re.Pattern]] = {}

# Raw patterns: intent_key → list of regex strings
# All patterns are case-insensitive and unicode-aware.
PATTERNS: dict[str, list[str]] = {
    "ack_started": [
        # Turkish
        r"^(tamam|peki|olur),?\s*(başl[ıi]yorum|bakıyorum|hemen)",
        r"^(başl[ıi]yorum|hemen\s+bakıyorum|üzerinde\s+çalışıyorum)",
        r"^bakalım",
        r"^hemen\s+hallediyorum",
        # English
        r"^(ok(ay)?|alright|sure),?\s*(let me|i'?ll|starting|on it)",
        r"^(on it|let me (work on|look into|check|handle))",
        r"^(i'?ll|let me)\s+(start|begin|get (on|to) (it|that))",
        r"^(working on it|getting (started|to it))",
    ],
    "ack_searching": [
        # Turkish
        r"(araştır[ıi]yorum|arat[ıi]yorum|arıyorum)",
        r"(bakıyorum|kontrol\s+ediyorum|inceliyorum)",
        r"^(bir\s+bak(ayım|alım))",
        # English
        r"(search(ing|ed)?(\s+for)?|look(ing)?\s+(up|into|for))",
        r"(finding|checking|researching|investigating)",
        r"^let me (search|look|find|check)",
    ],
    "ack_thinking": [
        # Turkish
        r"(düşünüyorum|düşüneyim|bir\s+saniye|bir\s+dakika)",
        r"^(hmm+|şey)",
        # English
        r"(think(ing|s)?(\s+about)?|consider(ing)?|ponder(ing)?)",
        r"^(hmm+|let me think|one (moment|second|sec))",
        r"^(give me a (moment|second|sec))",
    ],
    "ack_done": [
        # Turkish
        r"^(tamamlandı|bitti|hazır|işte|buyur(un)?)",
        r"^(hallettim|yaptım|bitirdim)",
        # English
        r"^(done|finished|completed?|here (you go|it is|are))",
        r"^(all (done|set|finished))",
        r"^(that'?s (done|it|all))",
        r"^(there you go|got it done)",
    ],
    "ack_error": [
        # Turkish
        r"(hata\s+(aldım|oluştu|var)|başarısız|yapamadım|sorun\s+(var|oluştu))",
        r"^(üzgünüm|maalesef|ne\s+yazık\s+ki)",
        # English
        r"(error|failed|couldn'?t|unable to|problem|issue)",
        r"^(sorry|unfortunately|i (couldn'?t|wasn'?t able|failed))",
        r"(went wrong|didn'?t work|broke|crash)",
    ],
    "ack_progress": [
        # Turkish
        r"(\d+[./]\d+\s*(tamamlandı|bitti|yapıldı))",
        r"(devam\s+ediyorum|ilerliyorum|sıradaki|adım\s*\d+)",
        r"(yarısı\s+(bitti|tamam)|yarıya\s+geldim)",
        # English
        r"(\d+\s*(of|/)\s*\d+\s*(done|complete|finished)?)",
        r"(step\s+\d+|progress|moving on|next (step|up))",
        r"(halfway|almost (done|there|finished))",
    ],
}


def _compile_patterns() -> None:
    """Compile all regex patterns once."""
    if _compiled:
        return
    for key, raw_list in PATTERNS.items():
        _compiled[key] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in raw_list
        ]


def match(text: str) -> Optional[str]:
    """Match text against all patterns.

    Returns the intent key (e.g. "ack_started") on first match, or None.
    Patterns are checked in definition order — put higher-priority intents first.
    """
    _compile_patterns()
    text = text.strip()
    if not text:
        return None
    for key, patterns in _compiled.items():
        for pattern in patterns:
            if pattern.search(text):
                return key
    return None


def add_patterns(key: str, new_patterns: list[str]) -> None:
    """Add custom patterns for an intent key at runtime.

    Useful for user-defined patterns loaded from config.
    """
    if key not in PATTERNS:
        PATTERNS[key] = []
    PATTERNS[key].extend(new_patterns)
    # Invalidate compiled cache for this key
    if key in _compiled:
        _compiled[key] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in PATTERNS[key]
        ]
    elif _compiled:
        # Other keys already compiled, compile this one too
        _compiled[key] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in PATTERNS[key]
        ]
