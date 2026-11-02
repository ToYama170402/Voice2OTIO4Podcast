import io

import pytest

from podcastpreprocessor.cli import _ProgressDisplay
from podcastpreprocessor import factory


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_progress_display_renders_tty_bar() -> None:
    stream = TTYBuffer()

    _ProgressDisplay(stream)("/audio/episode.wav", 5.0, 10.0)

    assert "Transcribing episode.wav:" in stream.getvalue()
    assert "50.0%" in stream.getvalue()


def test_progress_display_is_quiet_for_non_tty() -> None:
    stream = io.StringIO()

    _ProgressDisplay(stream)("episode.wav", 5.0, 10.0)

    assert stream.getvalue() == ""


def test_build_transcriber_keeps_whisper_as_default_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWhisperTranscriber:
        def __init__(self, *args: object) -> None:
            self.args = args

    monkeypatch.setattr(factory, "FasterWhisperTranscriber", FakeWhisperTranscriber)
    transcriber = factory.build_transcriber(
        "faster-whisper", "small", "cpu", "int8", "ignored", "ja", True, 60
    )

    assert isinstance(transcriber, FakeWhisperTranscriber)
    assert transcriber.args == ("small", "cpu", "int8")


def test_build_transcriber_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported transcription backend"):
        factory.build_transcriber(
            "unknown", "small", "cpu", "int8", "ignored", "ja", True, 60
        )
