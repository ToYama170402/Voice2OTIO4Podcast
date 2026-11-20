"""Overall progress reporting across files and processing phases.

The old display showed only the per-file transcription ratio, so the bar
reset for every input file and froze during work that happens outside the
transcription callbacks (SenseVoice audio loading / resampling / VAD, and
the final timeline generation). This reporter maps every phase onto one
overall ratio:

- each input file owns an equal slice of the transcription block (90% of
  the whole), split into preparation (15% of the slice) and transcription
  (85%, driven by the existing per-file progress callbacks);
- timeline generation (segment processing + OTIO/SRT writing) is the
  final 10%.

The ratio never decreases, and nothing is written unless the stream is a
TTY. The mapping itself is pure arithmetic so it can be unit tested
without a terminal.
"""

from pathlib import Path
from typing import TextIO

PHASE_TRANSCRIBE = "文字起こし"
PHASE_PROCESSING = "タイムライン生成"

TRANSCRIPTION_WEIGHT = 0.9
PROCESSING_WEIGHT = 1.0 - TRANSCRIPTION_WEIGHT
PREPARE_FRACTION = 0.15

_BAR_WIDTH = 30


class ProgressReporter:
    """Render one overall progress bar from per-file phase callbacks.

    Usable directly as the transcribers' ``progress_callback`` (called
    with ``(audio_path, elapsed, duration)``); preparation phases are
    reported through :meth:`phase`.
    """

    def __init__(self, stream: TextIO, total_files: int) -> None:
        if total_files <= 0:
            raise ValueError("total_files must be positive")
        self._stream = stream
        self._enabled = stream.isatty()
        self._total_files = total_files
        self._file_index = 0
        self._current_path: str | None = None
        self._phase_label = ""
        self._ratio = 0.0

    def ratio_for(self, file_index: int, phase: str, fraction: float) -> float:
        """Overall ratio for a point inside one file's slice.

        ``phase`` is ``"prepare"``, ``"transcribe"`` or ``"processing"``;
        ``fraction`` is the 0..1 progress inside that phase.
        """
        fraction = min(max(fraction, 0.0), 1.0)
        if phase == "processing":
            return TRANSCRIPTION_WEIGHT + PROCESSING_WEIGHT * fraction
        share = TRANSCRIPTION_WEIGHT / self._total_files
        base = file_index * share
        if phase == "prepare":
            return base + share * PREPARE_FRACTION * fraction
        if phase == "transcribe":
            return base + share * (
                PREPARE_FRACTION + (1.0 - PREPARE_FRACTION) * fraction
            )
        raise ValueError(f"unknown phase: {phase}")

    @property
    def ratio(self) -> float:
        return self._ratio

    def phase(self, audio_path: str, phase_label: str) -> None:
        """Report that a preparation phase started for ``audio_path``."""
        self._advance_file(audio_path)
        self._phase_label = phase_label
        self._set(self.ratio_for(self._file_index, "prepare", 0.0))

    def __call__(self, audio_path: str, elapsed: float, duration: float | None) -> None:
        """Transcription progress callback (elapsed/duration in seconds)."""
        self._advance_file(audio_path)
        self._phase_label = PHASE_TRANSCRIBE
        if duration is None or duration <= 0:
            self._render()
            return
        self._set(self.ratio_for(self._file_index, "transcribe", elapsed / duration))

    def start_processing(self) -> None:
        """Report the final timeline-generation phase."""
        self._phase_label = PHASE_PROCESSING
        self._set(self.ratio_for(self._file_index, "processing", 0.0))

    def finish(self) -> None:
        """Mark the whole run complete and close the bar line."""
        self._phase_label = "完了"
        self._ratio = 1.0
        self._render()
        if self._enabled:
            self._stream.write("\n")
            self._stream.flush()

    def _advance_file(self, audio_path: str) -> None:
        if self._current_path is None:
            self._current_path = audio_path
        elif audio_path != self._current_path:
            self._current_path = audio_path
            self._file_index = min(self._file_index + 1, self._total_files - 1)

    def _set(self, ratio: float) -> None:
        self._ratio = max(self._ratio, min(max(ratio, 0.0), 1.0))
        self._render()

    def _render(self) -> None:
        if not self._enabled:
            return
        filled = int(self._ratio * _BAR_WIDTH)
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        name = Path(self._current_path).name if self._current_path else ""
        counter = (
            f" ({self._file_index + 1}/{self._total_files})"
            if self._total_files > 1
            else ""
        )
        label = " ".join(part for part in (self._phase_label, name + counter) if part)
        self._stream.write(f"\r[{bar}] {self._ratio:>5.1%} {label}")
        self._stream.flush()
