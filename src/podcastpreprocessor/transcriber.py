"""Transcription backend contracts and the faster-whisper adapter."""

from collections.abc import Callable
from typing import Protocol

from .models import RawSegment, WordTimestamp

ProgressCallback = Callable[[str, float, float | None], None]
TranscriptionResult = tuple[str, str, list[RawSegment], float | None]


class Transcriber(Protocol):
    def transcribe(
        self, audio_path: str, source_id: str = "source_1"
    ) -> list[RawSegment]: ...

    def transcribe_sources(
        self,
        audio_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranscriptionResult]: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is required for transcription; install project dependencies first"
            ) from error
        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except (MemoryError, RuntimeError) as error:
            if _is_memory_error(error):
                raise RuntimeError(
                    "Whisperモデルをメモリに読み込めませんでした。"
                    " CPUでは --model small --compute-type int8 を試し、"
                    "まだ失敗する場合は --model base または tiny を指定してください。"
                ) from error
            raise

    def transcribe(
        self, audio_path: str, source_id: str = "source_1"
    ) -> list[RawSegment]:
        segments, info = self._model.transcribe(
            audio_path,
            language="ja",
            word_timestamps=True,
        )
        return [
            _to_raw_segment(index, segment, source_id)
            for index, segment in enumerate(segments)
        ]

    def transcribe_sources(
        self,
        audio_paths: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> list[TranscriptionResult]:
        """Transcribe one file per source and retain detected media durations."""
        results: list[TranscriptionResult] = []
        for index, audio_path in enumerate(audio_paths, start=1):
            source_id = f"source_{index}"
            segments, info = self._model.transcribe(
                audio_path,
                language="ja",
                word_timestamps=True,
            )
            duration = getattr(info, "duration", None)
            duration_value = float(duration) if duration else None
            if progress_callback is not None:
                progress_callback(audio_path, 0.0, duration_value)
            raw_segments: list[RawSegment] = []
            last_elapsed = 0.0
            for item_index, segment in enumerate(segments):
                raw_segments.append(_to_raw_segment(item_index, segment, source_id))
                last_elapsed = float(getattr(segment, "end"))
                if progress_callback is not None:
                    progress_callback(
                        audio_path,
                        last_elapsed,
                        duration_value,
                    )
            if progress_callback is not None and last_elapsed != duration_value:
                progress_callback(audio_path, duration_value or 0.0, duration_value)
            results.append(
                (
                    source_id,
                    audio_path,
                    raw_segments,
                    duration_value,
                )
            )
        return results


def _to_raw_segment(
    index: int, segment: object, source_id: str = "source_1"
) -> RawSegment:
    words: list[WordTimestamp] = []
    for word in getattr(segment, "words", None) or []:
        start = float(getattr(word, "start"))
        end = float(getattr(word, "end"))
        if end <= start:
            continue
        words.append(
            WordTimestamp(
                word=getattr(word, "word", ""),
                start=start,
                end=end,
                probability=float(getattr(word, "probability", 0.0)),
            )
        )
    return RawSegment(
        id=index,
        start=float(getattr(segment, "start")),
        end=float(getattr(segment, "end")),
        text=str(getattr(segment, "text", "")),
        no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0)),
        words=words,
        source_id=source_id,
    )


def _is_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("mkl_malloc", "failed to allocate memory", "out of memory")
    )
