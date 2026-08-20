"""Voice input via Groq's ``whisper-large-v3`` speech-to-text.

Speech recognition is clip-based: the browser records a short audio clip
(through ``st.audio_input``) and we send the bytes to Whisper, which returns the
transcript. There is no separate API key — voice reuses the same Groq client as
the rest of the app.

Text-to-speech (speaking answers aloud with the moving highlight) is handled in
the browser by the Web Speech API — see :mod:`src.voice_ui`. Whisper is
speech-to-text only.
"""

from __future__ import annotations

import logging

from src.llm import get_client, llm_available

log = logging.getLogger("talentsphere.voice")

WHISPER_MODEL = "whisper-large-v3"


def voice_available() -> bool:
    """Voice needs the same Groq key as the chat LLM."""
    return llm_available()


def transcribe(audio_bytes: bytes, filename: str = "speech.wav",
               prompt: str = "") -> str:
    """Transcribe recorded audio with Groq Whisper. Returns the text, or ""·

    Never raises: a transport or format problem returns an empty string so the
    caller can prompt the user to try again rather than crash the exam or chat.
    """
    if not audio_bytes:
        return ""
    client = get_client()
    if client is None:
        return ""
    try:
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=(filename, audio_bytes),
            # A short prompt biases Whisper toward domain vocabulary and cleaner
            # punctuation; harmless when empty.
            prompt=prompt or "Transcribe the speaker clearly.",
            temperature=0.0,
        )
        return (getattr(result, "text", "") or "").strip()
    except Exception:  # noqa: BLE001 - degrade to empty, let the UI re-prompt
        log.exception("whisper transcription failed")
        return ""
