"""faster-whisper adapter."""

from collections.abc import Iterable

from .models import RawSegment, WordTimestamp


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
        self, audio_paths: list[str]
    ) -> list[tuple[str, str, list[RawSegment], float | None]]:
        """Transcribe one file per source and retain detected media durations."""
        results: list[tuple[str, str, list[RawSegment], float | None]] = []
        for index, audio_path in enumerate(audio_paths, start=1):
            source_id = f"source_{index}"
            segments, info = self._model.transcribe(
                audio_path,
                language="ja",
                word_timestamps=True,
            )
            raw_segments = [
                _to_raw_segment(item_index, segment, source_id)
                for item_index, segment in enumerate(segments)
            ]
            duration = getattr(info, "duration", None)
            results.append(
                (
                    source_id,
                    audio_path,
                    raw_segments,
                    float(duration) if duration else None,
                )
            )
        return results


def _to_raw_segment(
    index: int, segment: object, source_id: str = "source_1"
) -> RawSegment:
    words: list[WordTimestamp] = []
    for word in getattr(segment, "words", None) or []:
        words.append(
            WordTimestamp(
                word=getattr(word, "word", ""),
                start=float(getattr(word, "start")),
                end=float(getattr(word, "end")),
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
