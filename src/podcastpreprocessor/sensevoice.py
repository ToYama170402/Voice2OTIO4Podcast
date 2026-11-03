"""sherpa-onnx SenseVoiceSmall transcription adapter."""

from importlib import import_module
import math
import re
from types import SimpleNamespace
from typing import Any

from .model_manager import SenseVoiceModelManager
from .models import RawSegment, WordTimestamp
from .transcriber import ProgressCallback, TranscriptionResult
from .vad import speech_segments_to_regions, vad_model_config

_TAG_PATTERN = re.compile(r"<\|[^|]+\|>")


class SenseVoiceTranscriber:
    def __init__(
        self,
        quantization: str = "int8",
        model_dir: str | None = None,
        cache_dir: str | None = None,
        offline: bool = False,
        provider: str = "cpu",
        num_threads: int = 4,
        language: str = "ja",
        use_itn: bool = True,
        chunk_duration_s: float = 30.0,
        vad: bool = False,
        vad_threshold_db: float = -40.0,
        vad_padding_s: float = 0.25,
        vad_model: str | None = None,
        vad_threshold: float = 0.5,
        vad_min_silence_duration: float = 0.25,
        vad_min_speech_duration: float = 0.1,
        vad_max_speech_duration: float = 120.0,
    ) -> None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be positive")
        if vad_padding_s < 0:
            raise ValueError("vad_padding_s must be non-negative")
        if not 0 < vad_threshold < 1:
            raise ValueError("vad_threshold must be between 0 and 1")
        if vad_min_silence_duration < 0 or vad_min_speech_duration < 0:
            raise ValueError("VAD durations must be non-negative")
        if vad_max_speech_duration <= 0:
            raise ValueError("vad_max_speech_duration must be positive")
        try:
            sherpa_onnx = import_module("sherpa_onnx")
            numpy = import_module("numpy")
            soundfile = import_module("soundfile")
        except ImportError as error:
            raise RuntimeError(
                "sherpa-onnx and soundfile are required for SenseVoice transcription; "
                "install the sensevoice extra first"
            ) from error
        model_path, tokens_path = SenseVoiceModelManager(cache_dir).resolve(
            quantization=quantization,
            model_dir=model_dir,
            offline=offline,
        )
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                provider=provider,
                num_threads=num_threads,
                language=language,
                use_itn=use_itn,
            )
        except (MemoryError, RuntimeError) as error:
            raise RuntimeError(
                "SenseVoiceSmallのONNXモデルを読み込めませんでした。"
                " --sensevoice-quantization int8 と --sensevoice-num-threads を確認してください。"
            ) from error
        self._soundfile = soundfile
        self._numpy = numpy
        self._sherpa_onnx = sherpa_onnx
        self._provider = provider
        self._num_threads = num_threads
        self._chunk_duration_s = chunk_duration_s
        self._vad = vad
        self._vad_threshold_db = vad_threshold_db
        self._vad_padding_s = vad_padding_s
        self._vad_model_path = (
            SenseVoiceModelManager(cache_dir).resolve_vad(vad_model, offline=offline)
            if vad
            else None
        )
        self._vad_threshold = vad_threshold
        self._vad_min_silence_duration = vad_min_silence_duration
        self._vad_min_speech_duration = vad_min_speech_duration
        self._vad_max_speech_duration = vad_max_speech_duration

    def transcribe(
        self, audio_path: str, source_id: str = "source_1"
    ) -> list[RawSegment]:
        return self._transcribe_one(audio_path, source_id)[0]

    def transcribe_sources(
        self,
        audio_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranscriptionResult]:
        results: list[TranscriptionResult] = []
        for index, audio_path in enumerate(audio_paths, start=1):
            source_id = f"source_{index}"
            segments, duration = self._transcribe_one(
                audio_path, source_id, progress_callback
            )
            results.append((source_id, audio_path, segments, duration))
        return results

    def _transcribe_one(
        self,
        audio_path: str,
        source_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[RawSegment], float]:
        audio, sample_rate = self._soundfile.read(
            audio_path, dtype="float32", always_2d=True
        )
        if len(audio) == 0 or sample_rate <= 0:
            raise RuntimeError(f"audio contains no samples: {audio_path}")
        audio = self._to_mono_16khz(audio, sample_rate)
        sample_rate = 16000
        duration = len(audio) / sample_rate
        chunk_samples = max(1, round(self._chunk_duration_s * sample_rate))
        segments: list[RawSegment] = []
        regions = (
            _detect_sherpa_speech_regions(
                self._sherpa_onnx,
                audio,
                sample_rate,
                str(self._vad_model_path),
                threshold=self._vad_threshold,
                min_silence_duration=self._vad_min_silence_duration,
                min_speech_duration=self._vad_min_speech_duration,
                max_speech_duration=self._vad_max_speech_duration,
                padding_s=self._vad_padding_s,
                provider=self._provider,
                num_threads=self._num_threads,
            )
            if self._vad
            else [(0, len(audio))]
        )
        if not regions:
            raise RuntimeError(f"VAD detected no speech: {audio_path}")
        if progress_callback is not None:
            progress_callback(audio_path, 0.0, duration)
        for region_start, region_end in regions:
            chunk_ranges = (
                [(region_start, region_end)]
                if self._vad
                else [
                    (chunk_start, min(chunk_start + chunk_samples, region_end))
                    for chunk_start in range(region_start, region_end, chunk_samples)
                ]
            )
            for chunk_start, chunk_end in chunk_ranges:
                offset = chunk_start / sample_rate
                chunk_duration = (chunk_end - chunk_start) / sample_rate
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sample_rate, audio[chunk_start:chunk_end])
                try:
                    self._recognizer.decode_stream(stream)
                except Exception as error:
                    raise RuntimeError(
                        "SenseVoiceSmallの推論に失敗しました。"
                        f" 音声の {offset:.1f}-{offset + chunk_duration:.1f} 秒区間です。"
                        " --sensevoice-chunk-duration-s を短くして再試行してください。"
                    ) from error
                result = stream.result
                if str(_value(result, "text", "")).strip():
                    segments.append(
                        _to_raw_segment(
                            result,
                            source_id,
                            offset,
                            offset + chunk_duration,
                            len(segments),
                        )
                    )
                if progress_callback is not None:
                    progress_callback(audio_path, offset + chunk_duration, duration)
        if not segments:
            raise RuntimeError(f"transcription produced no speech: {audio_path}")
        return segments, duration

    def _to_mono_16khz(self, audio: Any, sample_rate: int) -> Any:
        mono = audio[:, 0]
        if sample_rate == 16000:
            return mono
        target_length = round(len(mono) * 16000 / sample_rate)
        source_positions = self._numpy.arange(len(mono))
        target_positions = self._numpy.arange(target_length) * sample_rate / 16000
        return self._numpy.interp(target_positions, source_positions, mono).astype(
            "float32"
        )


def _to_raw_segment(
    result: object,
    source_id: str,
    offset: float,
    chunk_end: float,
    segment_id: int,
) -> RawSegment:
    text = _clean_text(str(_value(result, "text", "")))
    tokens = list(_value(result, "tokens", None) or [])
    timestamps = list(_value(result, "timestamps", None) or [])
    durations = list(_value(result, "durations", None) or [])
    if not text or not tokens or not timestamps:
        raise RuntimeError("SenseVoiceSmallの結果に単語タイムスタンプがありません。")
    if len(tokens) != len(timestamps):
        raise RuntimeError("SenseVoiceのtoken数とtimestamp数が一致しません")
    words: list[WordTimestamp] = []
    for index, (token, start_value) in enumerate(zip(tokens, timestamps, strict=True)):
        start = offset + float(start_value)
        if durations and len(durations) == len(tokens):
            end = start + float(durations[index])
        elif index + 1 < len(timestamps):
            end = offset + float(timestamps[index + 1])
        else:
            end = chunk_end
        token_text = _clean_text(str(token))
        if token_text and end > start:
            words.append(
                WordTimestamp(
                    word=token_text,
                    start=start,
                    end=end,
                    probability=1.0,
                )
            )
    if not words:
        raise RuntimeError("SenseVoiceSmallの単語タイムスタンプが空です")
    return RawSegment(
        id=segment_id,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        no_speech_prob=0.0,
        words=words,
        source_id=source_id,
    )


def _detect_sherpa_speech_regions(
    sherpa_onnx: Any,
    audio: Any,
    sample_rate: int,
    model_path: str,
    *,
    threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    max_speech_duration: float,
    padding_s: float,
    provider: str = "cpu",
    num_threads: int = 1,
) -> list[tuple[int, int]]:
    config = vad_model_config(
        sherpa_onnx,
        model_path,
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        min_speech_duration=min_speech_duration,
        max_speech_duration=max_speech_duration,
        window_size=512,
        provider=provider,
        num_threads=num_threads,
    )
    config.sample_rate = sample_rate
    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=max(30.0, max_speech_duration + 1.0),
    )
    window_size = config.silero_vad.window_size
    segments: list[object] = []
    for start in range(0, len(audio), window_size):
        vad.accept_waveform(audio[start : start + window_size])
        while not vad.empty():
            segment = vad.front
            segments.append(
                SimpleNamespace(
                    start=float(segment.start),
                    samples=segment.samples[:],
                )
            )
            vad.pop()
    vad.flush()
    while not vad.empty():
        segment = vad.front
        segments.append(
            SimpleNamespace(
                start=float(segment.start),
                samples=segment.samples[:],
            )
        )
        vad.pop()
    return speech_segments_to_regions(
        segments,
        sample_rate,
        len(audio),
        padding_s=padding_s,
    )


def _clean_text(value: str) -> str:
    return _TAG_PATTERN.sub("", value).strip()


def _detect_speech_regions(
    audio: Any,
    sample_rate: int,
    threshold_db: float = -40.0,
    frame_duration_s: float = 0.03,
    padding_s: float = 0.25,
) -> list[tuple[int, int]]:
    frame_samples = max(1, round(sample_rate * frame_duration_s))
    rms_values: list[float] = []
    for frame_start in range(0, len(audio), frame_samples):
        frame = audio[frame_start : frame_start + frame_samples]
        if len(frame) == 0:
            continue
        square_sum = math.fsum(float(sample) ** 2 for sample in frame)
        rms_values.append(math.sqrt(square_sum / len(frame)))
    if not rms_values:
        return []
    threshold = max(1e-5, max(rms_values) * 10 ** (threshold_db / 20.0))
    active_frames = [value >= threshold for value in rms_values]
    regions: list[tuple[int, int]] = []
    start_frame: int | None = None
    for index, active in enumerate(active_frames + [False]):
        if active and start_frame is None:
            start_frame = index
        elif not active and start_frame is not None:
            start = max(0, start_frame * frame_samples - round(padding_s * sample_rate))
            end = min(
                len(audio),
                index * frame_samples + round(padding_s * sample_rate),
            )
            if not regions or start > regions[-1][1]:
                regions.append((start, end))
            else:
                regions[-1] = (regions[-1][0], end)
            start_frame = None
    return regions


def _value(item: object, name: str, default: Any) -> Any:
    return getattr(item, name, default)
