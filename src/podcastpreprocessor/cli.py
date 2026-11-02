"""Command-line interface for PodcastPreprocessor."""

import argparse

from .pipeline import process_segments
from .transcriber import FasterWhisperTranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate FCP7 XML from Japanese audio"
    )
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("output", help="output FCP7 XML file")
    parser.add_argument("--model", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="auto", help="inference device")
    parser.add_argument(
        "--compute-type", default="default", help="faster-whisper compute type"
    )
    parser.add_argument("--fps", type=int, default=30, help="timeline frame rate")
    parser.add_argument("--sequence-name", default="Podcast_Timeline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        transcriber = FasterWhisperTranscriber(
            args.model, args.device, args.compute_type
        )
        raw_segments = transcriber.transcribe(args.audio)
    except RuntimeError as error:
        build_parser().error(str(error))
    process_segments(
        raw_segments, args.audio, args.output, args.fps, args.sequence_name
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
