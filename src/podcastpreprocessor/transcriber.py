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

    def transcribe(self, audio_path: str) -> list[RawSegment]:
        segments, _info = self._model.transcribe(
            audio_path,
            language="ja",
            word_timestamps=True,
        )
        return [
            _to_raw_segment(index, segment) for index, segment in enumerate(segments)
        ]


def _to_raw_segment(index: int, segment: object) -> RawSegment:
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
    )


def _is_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in ("mkl_malloc", "failed to allocate memory", "out of memory")
    )
