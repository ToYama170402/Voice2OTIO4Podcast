"""Speech-region utilities for the SenseVoice transcription adapter."""

from collections.abc import Iterable
from typing import Any


def speech_segments_to_regions(
    segments: Iterable[object],
    sample_rate: int,
    audio_length: int,
    padding_s: float = 0.25,
) -> list[tuple[int, int]]:
    """Convert sherpa VAD speech segments to padded sample ranges."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if audio_length < 0:
        raise ValueError("audio_length must be non-negative")
    if padding_s < 0:
        raise ValueError("padding_s must be non-negative")

    padding_samples = round(padding_s * sample_rate)
    regions: list[tuple[int, int]] = []
    for segment in segments:
        # sherpa-onnx SpeechSegment.start is an input-sample index, not seconds.
        start = round(float(getattr(segment, "start")))
        samples = getattr(segment, "samples", None)
        if samples is not None:
            end = start + len(samples)
        else:
            end = round(float(getattr(segment, "end")) * sample_rate)
        start = max(0, start - padding_samples)
        end = min(audio_length, end + padding_samples)
        if end <= start:
            continue
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return regions


def vad_model_config(
    sherpa_onnx: Any,
    model_path: str,
    *,
    threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    window_size: int,
    provider: str,
    num_threads: int,
) -> Any:
    """Build a sherpa-onnx Silero VAD configuration."""
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model_path
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence_duration
    config.silero_vad.min_speech_duration = min_speech_duration
    config.silero_vad.max_speech_duration = max_speech_duration
    config.silero_vad.window_size = window_size
    config.sample_rate = 16000
    config.num_threads = num_threads
    config.provider = provider
    return config
