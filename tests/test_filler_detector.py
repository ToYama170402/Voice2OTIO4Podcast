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


def test_filler_region_validation() -> None:
    import pytest

    from podcastpreprocessor.filler_detector import FillerRegion

    FillerRegion(0.0, 0.1)
    with pytest.raises(ValueError, match="filler region must satisfy"):
        FillerRegion(-0.1, 0.5)
    with pytest.raises(ValueError, match="filler region must satisfy"):
        FillerRegion(0.5, 0.5)
    with pytest.raises(ValueError, match="filler region must satisfy"):
        FillerRegion(0.5, 0.4)
    with pytest.raises(ValueError, match="filler region must satisfy"):
        FillerRegion(float("nan"), 0.5)
    with pytest.raises(ValueError, match="filler region must satisfy"):
        FillerRegion(0.0, float("inf"))


def test_find_islands_bridging_rules() -> None:
    from podcastpreprocessor.filler_detector import _find_islands

    def active(pattern: str) -> np.ndarray:
        return np.array([char == "1" for char in pattern])

    assert _find_islands(active(""), 1) == []
    assert _find_islands(active("000"), 1) == []
    assert _find_islands(active("11111"), 1) == [(0, 4)]
    assert _find_islands(active("00010"), 1) == [(3, 3)]
    # Gaps up to gap_frames are bridged; longer gaps split.
    assert _find_islands(active("101"), 1) == [(0, 2)]
    assert _find_islands(active("101"), 0) == [(0, 0), (2, 2)]
    assert _find_islands(active("1001"), 2) == [(0, 3)]
    assert _find_islands(active("1001"), 1) == [(0, 0), (3, 3)]
    assert _find_islands(active("10001"), 2) == [(0, 0), (4, 4)]
    # A trailing gap never extends the island past the last active frame.
    assert _find_islands(active("1100"), 5) == [(0, 1)]
    assert _find_islands(active("0110"), 1) == [(1, 2)]


def _island_inputs(
    n_frames: int,
    voiced_count: int,
    f0_value: float = 100.0,
    flux_value: float = 1.0,
):
    voiced = np.zeros(n_frames, dtype=bool)
    voiced[:voiced_count] = True
    f0 = np.full(n_frames, f0_value, dtype=np.float32)
    flux = np.full(n_frames, flux_value, dtype=np.float32)
    return voiced, f0, flux


def test_island_region_duration_boundaries() -> None:
    from podcastpreprocessor.filler_detector import _island_region

    voiced, f0, flux = _island_inputs(12, 12)
    region = _island_region(
        0, 11, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
    )
    assert region is not None
    assert (region.start, region.end) == (0.0, 0.12)

    voiced, f0, flux = _island_inputs(11, 11)
    assert (
        _island_region(
            0, 10, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
        )
        is None
    )

    voiced, f0, flux = _island_inputs(100, 100)
    region = _island_region(
        0, 99, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
    )
    assert region is not None
    assert region.end == 1.0

    voiced, f0, flux = _island_inputs(101, 101)
    assert (
        _island_region(
            0,
            100,
            voiced=voiced,
            f0=f0,
            flux=flux,
            hop_s=0.01,
            voiced_flux_median=1.0,
        )
        is None
    )


def test_island_region_voicing_rules() -> None:
    from podcastpreprocessor.filler_detector import _island_region

    # Exactly 45% voiced passes; less does not.
    voiced, f0, flux = _island_inputs(20, 9)
    assert (
        _island_region(
            0, 19, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
        )
        is not None
    )
    voiced, f0, flux = _island_inputs(20, 8)
    assert (
        _island_region(
            0, 19, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
        )
        is None
    )

    # Fewer than 4 voiced F0 frames is not enough evidence.
    voiced, f0, flux = _island_inputs(4, 2)
    assert (
        _island_region(
            0, 3, voiced=voiced, f0=f0, flux=flux, hop_s=0.05, voiced_flux_median=1.0
        )
        is None
    )


def test_island_region_stability_rules() -> None:
    from podcastpreprocessor.filler_detector import _island_region

    # Unstable F0 (alternating 100/200 Hz) exceeds the CV limit.
    voiced = np.ones(20, dtype=bool)
    f0 = np.array([100.0, 200.0] * 10, dtype=np.float32)
    flux = np.full(20, 1.0, dtype=np.float32)
    assert (
        _island_region(
            0, 19, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
        )
        is None
    )

    # Median flux above the voiced median rejects; equal passes.
    voiced, f0, flux = _island_inputs(20, 20, flux_value=2.0)
    assert (
        _island_region(
            0, 19, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=1.0
        )
        is None
    )
    assert (
        _island_region(
            0, 19, voiced=voiced, f0=f0, flux=flux, hop_s=0.01, voiced_flux_median=2.0
        )
        is not None
    )


def test_frame_features_shapes_and_estimates() -> None:
    from podcastpreprocessor.filler_detector import _frame_features

    # Too-short input still returns one (empty) frame slot per array.
    rms, flux, f0, peak, hop_s = _frame_features(np.zeros(100, dtype=np.float32), SR)
    assert len(rms) == len(flux) == len(f0) == len(peak) == 1
    assert hop_s == 0.01

    # Frame count follows 1 + (len - frame_len) // hop.
    _, _, _, _, _ = _frame_features(np.zeros(400, dtype=np.float32), SR)
    rms, _, _, _, _ = _frame_features(np.zeros(560, dtype=np.float32), SR)
    assert len(rms) == 2
    rms, _, _, _, _ = _frame_features(np.zeros(559, dtype=np.float32), SR)
    assert len(rms) == 1
    rms, _, _, _, _ = _frame_features(np.zeros(1600, dtype=np.float32), SR)
    assert len(rms) == 8

    # DC signal: RMS equals the constant level.
    rms, _, _, _, _ = _frame_features(np.full(1600, 0.5, dtype=np.float32), SR)
    assert np.allclose(rms, 0.5, atol=1e-5)

    # A steady 200 Hz sine is voiced with F0 near 200 Hz.
    _, _, f0, peak, _ = _frame_features(_tone(1.0, 200.0), SR)
    middle = slice(10, -10)
    assert np.all((f0[middle] > 180) & (f0[middle] < 220))
    assert np.all(peak[middle] > 0.45)


def test_detect_candidates_accepts_stereo_and_empty() -> None:
    stereo = np.stack([_silence(0.5), _silence(0.5)], axis=1)
    assert detect_candidates(stereo, SR) == []
    assert detect_candidates(np.zeros(0, dtype=np.float32), SR) == []
    assert detect_candidates(_silence(2.0), SR) == []


def test_token_filler_regions_sorts_and_joins_multi_char_fillers() -> None:
    from podcastpreprocessor.filler_detector import token_filler_regions

    later = _segment("ん", [("ん", 2.0, 2.3)])
    earlier = _segment(
        "えーと",
        [("え", 0.0, 0.1), ("ー", 0.1, 0.2), ("と", 0.2, 0.35)],
    )
    regions = token_filler_regions([later, earlier])
    assert [(r.start, r.end) for r in regions] == [(0.0, 0.35), (2.0, 2.3)]

    # Empty word tokens are ignored before tokenizing.
    with_empty = _segment(
        "あ",
        [("", 0.0, 0.0), ("あ", 0.1, 0.4)],
    )
    regions = token_filler_regions([with_empty])
    assert [(r.start, r.end) for r in regions] == [(0.1, 0.4)]
