"""Command-line interface for PodcastPreprocessor."""

import argparse
import os
from pathlib import Path
import sys

from .models import AudioSource, seconds_to_frame
from .pipeline import process_sources
from .factory import build_transcriber
from .filler_detector import detect_filler_candidates
from .progress import ProgressReporter
from .transcript_cache import default_cache_dir, transcribe_sources_cached

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
        "--sensevoice-audio-cache-dir",
        help="converted-audio cache directory (default: ~/.cache/podcastpreprocessor/converted)",
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
        "--sensevoice-vad-model",
        help="local Silero VAD ONNX model (default: download to cache)",
    )
    parser.add_argument(
        "--sensevoice-vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech probability threshold",
    )
    parser.add_argument(
        "--sensevoice-vad-min-silence-s",
        type=float,
        default=0.25,
        help="minimum silence duration used to end a speech region",
    )
    parser.add_argument(
        "--sensevoice-vad-min-speech-s",
        type=float,
        default=0.1,
        help="minimum speech duration to retain",
    )
    parser.add_argument(
        "--sensevoice-vad-max-speech-s",
        type=float,
        default=120.0,
        help="maximum duration of one VAD speech region",
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
    parser.add_argument(
        "--acoustic-fillers",
        action="store_true",
        help="detect filler candidates from the audio signal and mark them "
        "on the audio clips (works even when the transcription backend "
        "drops fillers, e.g. SenseVoice)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="transcript cache directory (default: ~/.cache/podcastpreprocessor/transcripts)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not reuse or store cached transcripts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        progress = ProgressReporter(sys.stderr, total_files=len(args.audio))
        settings = {
            "backend": args.backend,
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "language": args.language,
            "use_itn": args.use_itn,
            "batch_size_s": args.batch_size_s,
            "sensevoice_model": args.sensevoice_model,
            "sensevoice_quantization": args.sensevoice_quantization,
            "sensevoice_provider": args.sensevoice_provider,
            "sensevoice_num_threads": args.sensevoice_num_threads,
            "sensevoice_chunk_duration_s": args.sensevoice_chunk_duration_s,
            "sensevoice_vad": args.sensevoice_vad,
            "sensevoice_vad_threshold_db": args.sensevoice_vad_threshold_db,
            "sensevoice_vad_padding_s": args.sensevoice_vad_padding_s,
            "sensevoice_vad_model": args.sensevoice_vad_model,
            "sensevoice_vad_threshold": args.sensevoice_vad_threshold,
            "sensevoice_vad_min_silence_s": args.sensevoice_vad_min_silence_s,
            "sensevoice_vad_min_speech_s": args.sensevoice_vad_min_speech_s,
            "sensevoice_vad_max_speech_s": args.sensevoice_vad_max_speech_s,
        }
        cache_dir = None
        if not args.no_cache:
            cache_dir = (
                Path(args.cache_dir) if args.cache_dir else default_cache_dir()
            )

        def _build() -> object:
            return build_transcriber(
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
                args.sensevoice_vad_model,
                args.sensevoice_vad_threshold,
                args.sensevoice_vad_min_silence_s,
                args.sensevoice_vad_min_speech_s,
                args.sensevoice_vad_max_speech_s,
                sensevoice_audio_cache_dir=args.sensevoice_audio_cache_dir,
                sensevoice_transcript_cache_dir=(
                    str(cache_dir) if cache_dir is not None else None
                ),
            )

        def _on_cache_hit(audio_path: str) -> None:
            sys.stderr.write(
                f"Using cached transcript: {os.path.basename(audio_path)}\n"
            )

        transcribed = transcribe_sources_cached(
            _build,
            args.audio,
            settings,
            cache_dir,
            progress_callback=progress,
            phase_callback=progress.phase,
            on_cache_hit=_on_cache_hit,
        )
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
    filler_regions = None
    if args.acoustic_fillers:
        try:
            filler_regions = {}
            for source, _raw in sources:
                regions = detect_filler_candidates(source.path)
                filler_regions[source.source_id] = regions
                sys.stderr.write(
                    f"Acoustic filler candidates in "
                    f"{os.path.basename(source.path)}: {len(regions)}\n"
                )
        except RuntimeError as error:
            parser.error(str(error))
    try:
        progress.start_processing()
        process_sources(
            sources,
            args.output,
            args.fps,
            args.sequence_name,
            args.srt_output,
            filler_regions=filler_regions,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    progress.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
