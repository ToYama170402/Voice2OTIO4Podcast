"""Command-line interface for PodcastPreprocessor."""

import argparse
import os
import sys
from typing import TextIO

from .models import AudioSource, seconds_to_frame
from .pipeline import process_sources
from .transcriber import FasterWhisperTranscriber


class _ProgressDisplay:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._enabled = stream.isatty()

    def __call__(self, audio_path: str, elapsed: float, duration: float | None) -> None:
        if not self._enabled or duration is None or duration <= 0:
            return
        ratio = min(max(elapsed / duration, 0.0), 1.0)
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        name = os.path.basename(audio_path)
        self._stream.write(f"\rTranscribing {name}: [{bar}] {ratio:>6.1%}")
        if ratio >= 1.0:
            self._stream.write("\n")
        self._stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OpenTimelineIO from Japanese audio"
    )
    parser.add_argument("audio", nargs="+", help="input audio file(s), one per speaker")
    parser.add_argument("output", help="output OTIO file")
    parser.add_argument(
        "--srt-output",
        help="output SRT file (default: same path as output with .srt suffix)",
    )
    parser.add_argument("--model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="auto", help="inference device")
    parser.add_argument(
        "--compute-type", default="default", help="faster-whisper compute type"
    )
    parser.add_argument("--fps", type=int, default=30, help="timeline frame rate")
    parser.add_argument("--sequence-name", default="Podcast_Timeline")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        progress = _ProgressDisplay(sys.stderr)
        transcribed = FasterWhisperTranscriber(
            args.model, args.device, args.compute_type
        ).transcribe_sources(args.audio, progress_callback=progress)
    except RuntimeError as error:
        parser.error(str(error))
    sources: list[tuple[AudioSource, list]] = []
    for track_index, (source_id, audio_path, raw_segments, duration) in enumerate(
        transcribed, start=1
    ):
        end_frame = (
            seconds_to_frame(duration, args.fps)
            if duration
            else max(
                (seconds_to_frame(segment.end, args.fps) for segment in raw_segments),
                default=0,
            )
        )
        if end_frame <= 0:
            parser.error(f"transcription produced no duration: {audio_path}")
        sources.append(
            (AudioSource(source_id, audio_path, track_index, end_frame), raw_segments)
        )
    try:
        process_sources(
            sources,
            args.output,
            args.fps,
            args.sequence_name,
            args.srt_output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
