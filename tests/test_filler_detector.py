import numpy as np

from podcastpreprocessor.filler_detector import detect_candidates

SR = 16000


def _tone(seconds: float, freq: float, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (
        amp * np.sin(2 * np.pi * freq * t)
        + 0.5 * amp * np.sin(2 * np.pi * freq * 2 * t)
        + 0.25 * amp * np.sin(2 * np.pi * freq * 3 * t)
    ).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _chirp(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    freq = 100 + 500 * t / seconds
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (0.4 * np.sin(phase)).astype(np.float32)


def test_detects_steady_island_between_pauses() -> None:
    samples = np.concatenate(
        [_silence(1.0), _tone(0.5, 200.0), _silence(1.0), _chirp(2.0), _silence(1.0)]
    )

    regions = detect_candidates(samples, SR)

    assert len(regions) == 1
    assert regions[0].start < 1.2 and regions[0].end > 1.3


def test_rejects_long_steady_tone() -> None:
    samples = np.concatenate([_silence(0.5), _tone(3.0, 200.0), _silence(0.5)])

    assert detect_candidates(samples, SR) == []


def test_rejects_unvoiced_noise_island() -> None:
    rng = np.random.default_rng(42)
    noise = (0.3 * rng.standard_normal(int(0.5 * SR))).astype(np.float32)
    samples = np.concatenate([_silence(1.0), noise, _silence(1.0)])

    assert detect_candidates(samples, SR) == []
