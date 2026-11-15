"""Acoustic filler candidate detection, independent of transcription.

Some transcription backends (notably SenseVoice) drop filler utterances
from their output entirely, so text-based filler detection cannot see
them. Fillers are acoustically simple, though: they are typically uttered
as short standalone vocalizations between pauses, and their interior is a
single steady voiced sound. This module finds such regions with plain
signal processing (energy islands plus pitch/spectral stability) and
reports them as *candidates*. Detection is deliberately conservative and
imprecise compared to the text-based path: candidates are surfaced as
markers on the audio clips so an editor can audition and cut them by
hand, instead of becoming slices automatically.
"""

from dataclasses import dataclass
import math

import numpy as np

TARGET_SAMPLE_RATE = 16000
_FRAME_SECONDS = 0.025
_HOP_SECONDS = 0.010
_CHUNK_FRAMES = 6000

# Candidate shape: a short island of sound between pauses whose interior
# is mostly one steady voiced tone.
_MIN_ISLAND_SECONDS = 0.12
_MAX_ISLAND_SECONDS = 1.0
_MIN_PAUSE_SECONDS = 0.18
_MIN_VOICED_FRACTION = 0.45
_MAX_F0_CV = 0.28
_VOICING_PEAK = 0.45


@dataclass(frozen=True)
class FillerRegion:
    """A time range that acoustically looks like a filler utterance."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start)
            or not math.isfinite(self.end)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("filler region must satisfy 0 <= start < end")


def decode_audio_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode any supported audio file to mono float32 at 16 kHz (PyAV)."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "acoustic filler detection requires PyAV (installed with "
            "faster-whisper); install project dependencies first"
        ) from error
    container = av.open(audio_path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        raise RuntimeError(f"no audio stream found: {audio_path}")
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32), TARGET_SAMPLE_RATE
    samples = np.concatenate(chunks).astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples, TARGET_SAMPLE_RATE


def _frame_features(samples: np.ndarray, sample_rate: int):
    """Per-frame RMS, spectral flux, F0 and voicing peak (chunked)."""
    frame_len = int(_FRAME_SECONDS * sample_rate)
    hop = int(_HOP_SECONDS * sample_rate)
    n_frames = 1 + max(0, (len(samples) - frame_len) // hop)
    rms = np.zeros(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)
    f0 = np.zeros(n_frames, dtype=np.float32)
    peak = np.zeros(n_frames, dtype=np.float32)
    if n_frames <= 0 or len(samples) < frame_len:
        return rms, flux, f0, peak, hop / sample_rate
    window = np.hanning(frame_len).astype(np.float32)
    min_lag, max_lag = int(sample_rate / 400), int(sample_rate / 60)
    lags = np.arange(min_lag, max_lag)
    prev_logspec: np.ndarray | None = None
    for start_f in range(0, n_frames, _CHUNK_FRAMES):
        end_f = min(start_f + _CHUNK_FRAMES, n_frames)
        index = np.arange(start_f, end_f)
        frames = samples[index[:, None] * hop + np.arange(frame_len)[None, :]]
        rms[start_f:end_f] = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        spec = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-9
        logspec = np.log(spec)
        if prev_logspec is not None:
            flux[start_f] = np.sqrt(((logspec[0] - prev_logspec) ** 2).sum())
        if end_f - start_f > 1:
            flux[start_f + 1 : end_f] = np.sqrt(
                ((logspec[1:] - logspec[:-1]) ** 2).sum(axis=1)
            )
        prev_logspec = logspec[-1]
        padded = np.zeros((end_f - start_f, 1024), dtype=np.float32)
        padded[:, :frame_len] = frames * window
        spectrum = np.fft.rfft(padded, axis=1)
        acf = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :frame_len]
        acf0 = acf[:, 0] + 1e-9
        acf_win = acf[:, min_lag:max_lag] / (frame_len - lags)
        best = np.argmax(acf_win, axis=1)
        peak[start_f:end_f] = acf_win[np.arange(end_f - start_f), best] / (
            acf0 / frame_len
        )
        f0[start_f:end_f] = sample_rate / (best + min_lag)
    return rms, flux, f0, peak, hop / sample_rate


def detect_candidates(samples: np.ndarray, sample_rate: int) -> list[FillerRegion]:
    """Find short, steady, pause-bounded vocalizations in mono samples."""
    if samples.ndim > 1:
        samples = samples[:, 0]
    rms, flux, f0, peak, hop_s = _frame_features(samples, sample_rate)
    n_frames = len(rms)
    if n_frames == 0:
        return []
    voiced = peak > _VOICING_PEAK
    threshold = float(np.percentile(rms, 25)) * 1.6
    active = rms > threshold
    voiced_flux_median = float(np.median(flux[voiced])) if voiced.any() else 0.0
    gap_frames = max(1, int(_MIN_PAUSE_SECONDS / hop_s))

    islands: list[tuple[int, int]] = []
    index = 0
    while index < n_frames:
        if not active[index]:
            index += 1
            continue
        cursor = index
        gap = 0
        last = index
        while cursor + 1 < n_frames:
            if active[cursor + 1]:
                cursor += 1
                last = cursor
                gap = 0
            elif gap < gap_frames:
                gap += 1
                cursor += 1
            else:
                break
        islands.append((index, last))
        index = last + 1

    regions: list[FillerRegion] = []
    for start_f, end_f in islands:
        duration = (end_f - start_f + 1) * hop_s
        if not (_MIN_ISLAND_SECONDS <= duration <= _MAX_ISLAND_SECONDS):
            continue
        island_voiced = voiced[start_f : end_f + 1]
        if island_voiced.mean() < _MIN_VOICED_FRACTION:
            continue
        island_f0 = f0[start_f : end_f + 1][island_voiced]
        if len(island_f0) < 4:
            continue
        f0_cv = float(island_f0.std() / max(island_f0.mean(), 1e-6))
        median_flux = float(np.median(flux[start_f : end_f + 1]))
        if f0_cv <= _MAX_F0_CV and median_flux <= voiced_flux_median:
            regions.append(
                FillerRegion(start_f * hop_s, (end_f + 1) * hop_s)
            )
    return regions


def detect_filler_candidates(audio_path: str) -> list[FillerRegion]:
    """Decode an audio file and detect acoustic filler candidates."""
    samples, sample_rate = decode_audio_mono(audio_path)
    return detect_candidates(samples, sample_rate)
