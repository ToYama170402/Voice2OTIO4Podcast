from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from podcastpreprocessor.audio_io import prepare_audio

SR = 44100


def _write_wav(path: Path, samplerate: int, channels: int, seconds: float = 1.2) -> None:
    t = np.arange(int(seconds * samplerate)) / samplerate
    tone = 0.4 * np.sin(2 * np.pi * 220 * t)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(str(path), data, samplerate)


def test_prepare_converts_to_16khz_mono_and_caches(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, SR, channels=2)
    cache = tmp_path / "converted"

    prepared = prepare_audio(str(source), cache_dir=cache)

    assert prepared != str(source)
    info = sf.info(prepared)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert abs(info.duration - 1.2) < 0.05
    samples, _ = sf.read(prepared, dtype="float32")
    assert np.max(np.abs(samples)) > 0.1

    # Second call reuses the converted file.
    mtime = Path(prepared).stat().st_mtime_ns
    assert prepare_audio(str(source), cache_dir=cache) == prepared
    assert Path(prepared).stat().st_mtime_ns == mtime


def test_prepare_keeps_ready_audio_untouched(tmp_path: Path) -> None:
    source = tmp_path / "ready.wav"
    _write_wav(source, 16000, channels=1)
    cache = tmp_path / "converted"

    assert prepare_audio(str(source), cache_dir=cache) == str(source)
    assert not cache.exists()


def test_prepare_converts_m4a(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    source = tmp_path / "input.m4a"
    container = av.open(str(source), mode="w")
    stream = container.add_stream("aac", rate=SR)
    stream.layout = "stereo"
    samples = int(1.2 * SR)
    tone = 0.4 * np.sin(2 * np.pi * 220 * np.arange(samples) / SR)
    frame = av.AudioFrame.from_ndarray(
        np.stack([tone, tone]).astype(np.float32),
        format="fltp",
        layout="stereo",
    )
    frame.sample_rate = SR
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    prepared = prepare_audio(str(source), cache_dir=tmp_path / "converted")

    info = sf.info(prepared)
    assert info.samplerate == 16000
    assert info.channels == 1
    assert abs(info.duration - 1.2) < 0.2
