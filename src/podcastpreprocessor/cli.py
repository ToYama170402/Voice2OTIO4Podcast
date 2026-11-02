"""Command-line interface for PodcastPreprocessor."""

import argparse
import os
import sys
from typing import TextIO

from .models import AudioSource, seconds_to_frame
from .pipeline import process_sources
from .factory import build_transcriber


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
        "--backend",
        choices=("faster-whisper", "sensevoice"),
        default="faster-whisper",
        help="transcription backend",
    )
    parser.add_argument(
        "--srt-output",
        help="output SRT file (default: same path as output with .srt suffix)",
    )
    parser.add_argument("--model", default="small", help="faster-whisper model size")
    parser.add_argument(
        "--sensevoice-model",
        default=None,
        help="local SenseVoice model directory (default: download to cache)",
    )
    parser.add_argument(
        "--sensevoice-quantization",
        choices=("none", "int8", "int4"),
        default="int8",
        help="SenseVoice ONNX model variant",
    )
    parser.add_argument(
        "--sensevoice-cache-dir", help="SenseVoice model cache directory"
    )
    parser.add_argument(
        "--sensevoice-num-threads",
        type=int,
        default=4,
        help="SenseVoice CPU inference thread count",
    )
    parser.add_argument(
        "--sensevoice-provider",
        choices=("cpu", "cuda", "coreml"),
        default="cpu",
        help="SenseVoice ONNX Runtime provider",
    )
    parser.add_argument(
        "--sensevoice-chunk-duration-s",
        type=float,
        default=30.0,
        help="SenseVoice audio chunk duration in seconds",
    )
    parser.add_argument(
        "--sensevoice-vad",
        action="store_true",
        help="skip non-speech regions with lightweight energy VAD",
    )
    parser.add_argument(
        "--sensevoice-vad-threshold-db",
        type=float,
        default=-40.0,
        help="VAD threshold relative to the loudest frame in dB",
    )
    parser.add_argument(
        "--sensevoice-vad-padding-s",
        type=float,
        default=0.25,
        help="speech padding added around VAD regions in seconds",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download a missing SenseVoice model",
    )
    parser.add_argument("--language", default="ja", help="SenseVoice language code")
    parser.add_argument(
        "--use-itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable SenseVoice inverse text normalization",
    )
    parser.add_argument(
        "--batch-size-s",
        type=int,
        default=60,
        help="SenseVoice batch duration in seconds",
    )
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
        transcribed = build_transcriber(
            args.backend,
            args.model,
            args.device,
            args.compute_type,
            args.sensevoice_model,
            args.language,
            args.use_itn,
            args.batch_size_s,
            args.sensevoice_quantization,
            args.sensevoice_model,
            args.sensevoice_cache_dir,
            args.offline,
            args.sensevoice_provider,
            args.sensevoice_num_threads,
            args.sensevoice_chunk_duration_s,
            args.sensevoice_vad,
            args.sensevoice_vad_threshold_db,
            args.sensevoice_vad_padding_s,
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
