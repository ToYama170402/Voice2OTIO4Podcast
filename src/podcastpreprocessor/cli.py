"""Command-line interface for PodcastPreprocessor."""

import argparse

from .models import AudioSource, seconds_to_frame
from .pipeline import process_sources
from .transcriber import FasterWhisperTranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate OpenTimelineIO from Japanese audio"
    )
    parser.add_argument("audio", nargs="+", help="input audio file(s), one per speaker")
    parser.add_argument("output", help="output OTIO file")
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
        transcribed = FasterWhisperTranscriber(
            args.model, args.device, args.compute_type
        ).transcribe_sources(args.audio)
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
        process_sources(sources, args.output, args.fps, args.sequence_name)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
