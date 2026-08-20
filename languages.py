"""Multi-language support for the voice assistant.

Scoped to the four supported languages: English, Hindi, Telugu, Tamil. Maps
each to a BCP-47 locale (used by the browser's speech recognition + text-to-
speech), provides localized welcome/confirmation strings, and detects spoken
"switch language" commands so a user can change language hands-free.
"""

from __future__ import annotations

import re

# Language name → BCP-47 code. Order defines the UI order.
LANGUAGES: dict[str, str] = {
    "english": "en-US",
    "hindi": "hi-IN",
    "telugu": "te-IN",
    "tamil": "ta-IN",
}

_CODE_TO_NAME = {code: name.title() for name, code in LANGUAGES.items()}

# Localized welcome — spoken the moment voice mode starts.
_WELCOME: dict[str, str] = {
    "en-US": ("Welcome {name}. I'm your Talent Sphere voice assistant. "
              "How can I help you today?"),
    "hi-IN": ("नमस्ते {name}। मैं आपका टैलेंट स्फियर वॉइस असिस्टेंट हूँ। "
              "मैं आपकी कैसे मदद कर सकता हूँ?"),
    "te-IN": ("నమస్కారం {name}. నేను మీ టాలెంట్ స్ఫియర్ వాయిస్ అసిస్టెంట్‌ని. "
              "నేను మీకు ఎలా సహాయం చేయగలను?"),
    "ta-IN": ("வணக்கம் {name}. நான் உங்கள் டேலன்ட் ஸ்பியர் குரல் உதவியாளர். "
              "இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?"),
}

# Localized confirmation when the language is switched.
_SWITCH_CONFIRM: dict[str, str] = {
    "en-US": "Sure — I'll continue in English.",
    "hi-IN": "ठीक है — अब मैं हिंदी में बात करूँगा।",
    "te-IN": "సరే — ఇప్పటి నుండి తెలుగులో మాట్లాడతాను.",
    "ta-IN": "சரி — இனி தமிழில் பேசுகிறேன்.",
}

# Phrases that signal a language switch, e.g. "switch to Hindi", "talk in Tamil".
_SWITCH_HINT = re.compile(
    r"\b(switch|change|talk|speak|respond|reply|convert|continue)\b", re.IGNORECASE
)


def language_name(code: str) -> str:
    """Human name for a locale code ('hi-IN' → 'Hindi')."""
    return _CODE_TO_NAME.get(code, code)


def default_code() -> str:
    return "en-US"


def welcome_text(code: str, name: str) -> str:
    return _WELCOME.get(code, _WELCOME["en-US"]).format(name=name)


def switch_confirm(code: str) -> str:
    return _SWITCH_CONFIRM.get(code, _SWITCH_CONFIRM["en-US"])


def detect_language_switch(text: str) -> str | None:
    """Return a locale code if the text is a 'switch language' command, else None.

    Requires both a switch verb ('switch', 'change', 'speak in', …) and a known
    language name, so ordinary questions that merely mention a language aren't
    misread as a command.
    """
    if not text:
        return None
    low = text.lower()
    if not _SWITCH_HINT.search(low):
        return None
    for name, code in LANGUAGES.items():
        if re.search(rf"\b{name}\b", low):
            return code
    return None
