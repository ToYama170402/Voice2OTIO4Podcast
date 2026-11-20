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
    assert regions[0].start < 1.2
    assert regions[0].end > 1.3


def test_rejects_long_steady_tone() -> None:
    samples = np.concatenate([_silence(0.5), _tone(3.0, 200.0), _silence(0.5)])

    assert detect_candidates(samples, SR) == []


def test_rejects_unvoiced_noise_island() -> None:
    rng = np.random.default_rng(42)
    noise = (0.3 * rng.standard_normal(int(0.5 * SR))).astype(np.float32)
    samples = np.concatenate([_silence(1.0), noise, _silence(1.0)])

    assert detect_candidates(samples, SR) == []


def _segment(text: str, chars: list[tuple[str, float, float]]):
    from podcastpreprocessor.models import RawSegment, WordTimestamp

    return RawSegment(
        1,
        chars[0][1],
        chars[-1][2],
        text,
        0.0,
        [WordTimestamp(c, s, e, 1.0) for c, s, e in chars],
    )


def test_token_filler_regions_finds_standalone_filler() -> None:
    from podcastpreprocessor.filler_detector import token_filler_regions

    segment = _segment(
        "あ今日は晴れ",
        [
            ("あ", 0.0, 0.3),
            ("今", 0.3, 0.45),
            ("日", 0.45, 0.6),
            ("は", 0.6, 0.75),
            ("晴", 0.75, 0.95),
            ("れ", 0.95, 1.1),
        ],
    )
    regions = token_filler_regions([segment])
    assert len(regions) == 1
    assert (regions[0].start, regions[0].end) == (0.0, 0.3)


def test_token_filler_regions_ignores_nasal_inside_word() -> None:
    from podcastpreprocessor.filler_detector import token_filler_regions

    segment = _segment(
        "そんな日",
        [("そ", 0.0, 0.15), ("ん", 0.15, 0.3), ("な", 0.3, 0.45), ("日", 0.45, 0.6)],
    )
    assert token_filler_regions([segment]) == []


def test_token_filler_regions_skips_segments_without_words() -> None:
    from podcastpreprocessor.filler_detector import token_filler_regions
    from podcastpreprocessor.models import RawSegment

    assert token_filler_regions([RawSegment(1, 0.0, 1.0, "あ", 0.0, [])]) == []
