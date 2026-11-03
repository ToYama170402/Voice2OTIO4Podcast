import sys
from types import ModuleType, SimpleNamespace

import pytest

from podcastpreprocessor.sensevoice import (
    SenseVoiceTranscriber,
    _detect_sherpa_speech_regions,
    _detect_speech_regions,
)
from podcastpreprocessor.vad import speech_segments_to_regions, vad_model_config


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


def test_sensevoice_vad_region_is_not_split_into_fixed_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        text="こんにちは",
        tokens=["こ"],
        timestamps=[0.0],
        durations=[0.1],
    )
    _install_sherpa_mocks(monkeypatch, result)
    monkeypatch.setattr(
        "podcastpreprocessor.model_manager.SenseVoiceModelManager.resolve_vad",
        lambda *args, **kwargs: "silero_vad.onnx",
    )
    monkeypatch.setattr(
        "podcastpreprocessor.sensevoice._detect_sherpa_speech_regions",
        lambda *args, **kwargs: [(0, 16000)],
    )

    transcriber = SenseVoiceTranscriber(vad=True, chunk_duration_s=0.3)
    segments = transcriber.transcribe("episode.wav")

    assert len(segments) == 1
    assert segments[0].start == 0.0


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


def test_speech_segments_to_regions_uses_samples_and_merges_padding() -> None:
    segments = [
        SimpleNamespace(start=1000, samples=[0.0] * 100),
        SimpleNamespace(start=1150, samples=[0.0] * 100),
    ]

    regions = speech_segments_to_regions(
        segments,
        sample_rate=1000,
        audio_length=3000,
        padding_s=0.1,
    )

    assert regions == [(900, 1350)]


def test_speech_segments_to_regions_clamps_audio_bounds() -> None:
    segments = [SimpleNamespace(start=0.05, samples=[0.0] * 50)]

    regions = speech_segments_to_regions(
        segments,
        sample_rate=1000,
        audio_length=100,
        padding_s=0.25,
    )

    assert regions == [(0, 100)]


def test_vad_model_config_sets_silero_options() -> None:
    class Config:
        def __init__(self) -> None:
            self.silero_vad = SimpleNamespace(window_size=512)

    class FakeSherpa:
        VadModelConfig = Config

    config = vad_model_config(
        FakeSherpa,
        "silero_vad.onnx",
        threshold=0.6,
        min_silence_duration=0.3,
        min_speech_duration=0.1,
        max_speech_duration=60.0,
        window_size=256,
        provider="cpu",
        num_threads=2,
    )

    assert config.silero_vad.model == "silero_vad.onnx"
    assert config.silero_vad.window_size == 256
    assert config.sample_rate == 16000
    assert config.num_threads == 2


def test_detect_sherpa_speech_regions_feeds_windows_and_flushes() -> None:
    class FakeConfig:
        def __init__(self) -> None:
            self.silero_vad = SimpleNamespace(window_size=2)

    class FakeVad:
        def __init__(self, config: object, buffer_size_in_seconds: float) -> None:
            self.segments = [SimpleNamespace(start=0.0, samples=[1.0, 1.0])]
            self.accepted: list[object] = []
            self.flushed = False

        def accept_waveform(self, samples: object) -> None:
            self.accepted.append(samples)

        def empty(self) -> bool:
            return not self.segments

        @property
        def front(self) -> object:
            return self.segments[0]

        def pop(self) -> None:
            self.segments.pop(0)

        def flush(self) -> None:
            self.flushed = True

    class FakeSherpa:
        VadModelConfig = FakeConfig
        VoiceActivityDetector = FakeVad

    regions = _detect_sherpa_speech_regions(
        FakeSherpa,
        [0.0] * 5,
        2,
        "silero_vad.onnx",
        threshold=0.5,
        min_silence_duration=0.25,
        min_speech_duration=0.1,
        max_speech_duration=60.0,
        padding_s=0.0,
    )

    assert regions == [(0, 2)]
