from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from podcastpreprocessor.audio_io import (
    _file_key,
    list_audio_streams,
    prepare_audio,
)

SR = 44100


def write_two_stream_mka(
    path: Path, freq_a: float = 200.0, freq_b: float = 600.0, seconds: float = 1.0
) -> Path:
    """Write a Matroska file holding two stereo FLAC audio streams."""
    import av

    rate = 48000
    total = int(rate * seconds)
    container = av.open(str(path), "w")
    streams = [container.add_stream("flac", rate=rate) for _ in (freq_a, freq_b)]
    for stream, freq in zip(streams, (freq_a, freq_b), strict=True):
        done = 0
        while done < total:
            count = min(4096, total - done)
            t = np.arange(done, done + count) / rate
            mono = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
            interleaved = np.empty(count * 2, dtype=np.int16)
            interleaved[0::2] = mono
            interleaved[1::2] = mono
            frame = av.AudioFrame(format="s16", layout="stereo", samples=count)
            frame.sample_rate = rate
            frame.pts = done
            frame.planes[0].update(interleaved.tobytes())
            for packet in stream.encode(frame):
                container.mux(packet)
            done += count
        for packet in stream.encode(None):
            container.mux(packet)
    container.close()
    return path


def _dominant_frequency(path: str) -> float:
    data, samplerate = sf.read(path)
    spectrum = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(data), 1 / samplerate)
    return float(freqs[int(np.argmax(spectrum))])


def _write_wav(
    path: Path, samplerate: int, channels: int, seconds: float = 1.2
) -> None:
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


def test_prepare_resumes_interrupted_conversion(tmp_path: Path) -> None:
    import json

    from podcastpreprocessor.audio_io import _file_key, _partial_paths

    source = tmp_path / "input.wav"
    _write_wav(source, SR, channels=2, seconds=3.0)
    reference = prepare_audio(str(source), cache_dir=tmp_path / "full")
    ref_samples, _ = sf.read(reference, dtype="float32")

    cache = tmp_path / "resumed"
    cache.mkdir()
    dest = cache / f"{_file_key(str(source))}.wav"
    partial, journal = _partial_paths(dest)
    ref_int16, _ = sf.read(reference, dtype="int16")
    partial.write_bytes(ref_int16[:16000].tobytes())
    journal.write_text(json.dumps({"version": 1, "samples": 16000}))

    prepared = prepare_audio(str(source), cache_dir=cache)

    assert prepared == str(dest)
    samples, _ = sf.read(prepared, dtype="float32")
    assert abs(len(samples) - len(ref_samples)) < 800  # within 50 ms
    # The tone is continuous across the resume seam and to the end.
    for start_s in (0.95, 2.5):
        window = samples[int(start_s * 16000) : int((start_s + 0.1) * 16000)]
        assert float(np.sqrt(np.mean(window**2))) > 0.1
    # Partials are cleaned up after a successful conversion.
    assert not partial.exists()
    assert not journal.exists()


def test_prepare_restarts_when_partial_is_inconsistent(tmp_path: Path) -> None:
    import json

    from podcastpreprocessor.audio_io import _file_key, _partial_paths

    source = tmp_path / "input.wav"
    _write_wav(source, SR, channels=2, seconds=1.2)
    cache = tmp_path / "converted"
    cache.mkdir()
    dest = cache / f"{_file_key(str(source))}.wav"
    partial, journal = _partial_paths(dest)
    partial.write_bytes(b"\x00" * 100)  # far less than the journal claims
    journal.write_text(json.dumps({"version": 1, "samples": 16000}))

    prepared = prepare_audio(str(source), cache_dir=cache)

    info = sf.info(prepared)
    assert info.samplerate == 16000
    assert abs(info.duration - 1.2) < 0.05


def test_list_audio_streams_reports_each_track(tmp_path: Path) -> None:
    source = write_two_stream_mka(tmp_path / "two.mka")

    streams = list_audio_streams(str(source))

    assert [(s.stream_index, s.channels, s.sample_rate) for s in streams] == [
        (0, 2, 48000),
        (1, 2, 48000),
    ]


def test_prepare_audio_decodes_the_requested_stream(tmp_path: Path) -> None:
    source = write_two_stream_mka(tmp_path / "two.mka")
    cache = tmp_path / "converted"

    first = prepare_audio(str(source), cache_dir=cache, stream_index=0)
    second = prepare_audio(str(source), cache_dir=cache, stream_index=1)

    assert first != second
    assert _dominant_frequency(first) == pytest.approx(200.0, abs=5.0)
    assert _dominant_frequency(second) == pytest.approx(600.0, abs=5.0)


def test_stream_zero_keeps_the_historical_cache_key(tmp_path: Path) -> None:
    source = write_two_stream_mka(tmp_path / "two.mka")

    assert _file_key(str(source)) == _file_key(str(source), 0)
    assert _file_key(str(source), 1) != _file_key(str(source), 0)


def test_prepare_audio_rejects_a_missing_stream(tmp_path: Path) -> None:
    source = write_two_stream_mka(tmp_path / "two.mka")

    with pytest.raises(RuntimeError, match="audio stream 2 not found"):
        prepare_audio(str(source), cache_dir=tmp_path / "c", stream_index=2)
