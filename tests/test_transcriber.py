import sys
from types import ModuleType, SimpleNamespace

import pytest

from podcastpreprocessor.transcriber import FasterWhisperTranscriber


def test_transcribe_sources_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    faster_whisper = ModuleType("faster_whisper")

    class FakeWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def transcribe(
            self, audio_path: str, **kwargs: object
        ) -> tuple[object, object]:
            segments = [
                SimpleNamespace(start=0.0, end=4.0, text="one"),
                SimpleNamespace(start=4.0, end=10.0, text="two"),
            ]
            return segments, SimpleNamespace(duration=10.0)

    faster_whisper.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    progress: list[tuple[str, float, float | None]] = []

    transcriber = FasterWhisperTranscriber()
    results = transcriber.transcribe_sources(
        ["episode.wav"], progress_callback=lambda *values: progress.append(values)
    )

    assert len(results[0][2]) == 2
    assert progress == [
        ("episode.wav", 0.0, 10.0),
        ("episode.wav", 4.0, 10.0),
        ("episode.wav", 10.0, 10.0),
    ]


def test_transcriber_discards_zero_duration_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faster_whisper = ModuleType("faster_whisper")

    class FakeWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def transcribe(
            self, audio_path: str, **kwargs: object
        ) -> tuple[object, object]:
            segments = [
                SimpleNamespace(
                    start=0.0,
                    end=1.0,
                    text="ok",
                    words=[
                        SimpleNamespace(word="", start=0.2, end=0.2, probability=1.0),
                        SimpleNamespace(word="ok", start=0.3, end=0.8, probability=1.0),
                    ],
                )
            ]
            return segments, SimpleNamespace(duration=1.0)

    faster_whisper.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    segments = FasterWhisperTranscriber().transcribe("episode.wav")

    assert [word.word for word in segments[0].words] == ["ok"]


def test_transcriber_explains_cpu_memory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faster_whisper = ModuleType("faster_whisper")

    class FailingWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("mkl_malloc: failed to allocate memory")

    faster_whisper.WhisperModel = FailingWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    with pytest.raises(RuntimeError, match="--model small --compute-type int8"):
        FasterWhisperTranscriber("turbo", "cpu", "int8")


def test_transcriber_preserves_non_memory_model_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faster_whisper = ModuleType("faster_whisper")

    class FailingWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("invalid compute type")

    faster_whisper.WhisperModel = FailingWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)

    with pytest.raises(RuntimeError, match="invalid compute type"):
        FasterWhisperTranscriber("small", "cpu", "bad")
