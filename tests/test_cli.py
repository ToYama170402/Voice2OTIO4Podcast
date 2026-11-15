import pytest

from podcastpreprocessor import factory


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
