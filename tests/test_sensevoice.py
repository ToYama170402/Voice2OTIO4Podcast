import sys
from types import ModuleType, SimpleNamespace

import pytest

from podcastpreprocessor.sensevoice import (
    SenseVoiceTranscriber,
    _detect_speech_regions,
)


class FakeAudio:
    def __len__(self) -> int:
        return 1000

    def __getitem__(self, key: object) -> list[float]:
        return [0.0] * 1000


def _install_sherpa_mocks(monkeypatch: pytest.MonkeyPatch, result: object) -> None:
    sherpa_onnx = ModuleType("sherpa_onnx")
    soundfile = ModuleType("soundfile")

    class FakeStream:
        def __init__(self) -> None:
            self.result = result

        def accept_waveform(self, sample_rate: int, audio: object) -> None:
            pass

    class FakeRecognizer:
        @classmethod
        def from_sense_voice(cls, **kwargs: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            pass

    sherpa_onnx.OfflineRecognizer = FakeRecognizer  # type: ignore[attr-defined]
    soundfile.read = lambda *args, **kwargs: (FakeAudio(), 1000)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa_onnx)
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setattr(
        "podcastpreprocessor.sensevoice.SenseVoiceModelManager.resolve",
        lambda *args, **kwargs: ("model.int8.onnx", "tokens.txt"),
    )


def test_sensevoice_transcribes_timestamped_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        text="<|ja|><|NEUTRAL|>こんにちは",
        tokens=list("こんにちは"),
        timestamps=[0.0, 0.2, 0.4, 0.6, 0.8],
        durations=[0.2, 0.2, 0.2, 0.2, 0.2],
    )
    _install_sherpa_mocks(monkeypatch, result)

    transcriber = SenseVoiceTranscriber()
    results = transcriber.transcribe_sources(["episode.wav"])

    segments = results[0][2]
    assert results[0][0] == "source_1"
    assert results[0][3] == 1.0
    assert segments[0].text == "こんにちは"
    assert [word.word for word in segments[0].words] == list("こんにちは")
    assert segments[0].words[1].start == 0.2
    assert segments[0].words[-1].end == 1.0


def test_sensevoice_requires_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sherpa_mocks(monkeypatch, SimpleNamespace(text="こんにちは"))

    with pytest.raises(RuntimeError, match="単語タイムスタンプ"):
        SenseVoiceTranscriber().transcribe("episode.wav")


def test_sensevoice_adds_chunk_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        text="こんにちは",
        tokens=["こ"],
        timestamps=[0.0],
        durations=[0.1],
    )
    _install_sherpa_mocks(monkeypatch, result)

    transcriber = SenseVoiceTranscriber(chunk_duration_s=0.3)
    segments = transcriber.transcribe("episode.wav")

    assert [segment.start for segment in segments] == [0.0, 0.3, 0.6, 0.9]


def test_detect_speech_regions_skips_silence() -> None:
    audio = [0.0] * 100 + [1.0] * 100 + [0.0] * 100

    regions = _detect_speech_regions(
        audio,
        sample_rate=100,
        threshold_db=-20.0,
        frame_duration_s=0.1,
        padding_s=0.0,
    )

    assert regions == [(100, 200)]
