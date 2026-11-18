"""Command-line interface for PodcastPreprocessor."""

import argparse
import sys
from pathlib import Path

from .factory import build_transcriber
from .filler_detector import FillerRegion, detect_filler_candidates
from .models import AudioSource, RawSegment, seconds_to_frame
from .pipeline import process_sources
from .progress import ProgressReporter
from .transcriber import Transcriber
from .transcript_cache import default_cache_dir, transcribe_sources_cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-preprocessor",
        description=(
            "日本語ポッドキャストの音声を文字起こしし、字幕・カット候補・"
            "元音声をリンクしたOpenTimelineIO（.otio）を生成します。"
            "生成したOTIOはDaVinci Resolveに読み込んで編集できます。"
        ),
        epilog=(
            "使用例:\n"
            "  最小の実行（faster-whisper、話者1人）:\n"
            "    podcast-preprocessor speaker_a.wav podcast.otio\n"
            "\n"
            "  話者2人、CPU向けの設定:\n"
            "    podcast-preprocessor speaker_a.wav speaker_b.wav podcast.otio"
            " --model small --device cpu --compute-type int8\n"
            "\n"
            "  SenseVoiceを使う:\n"
            "    podcast-preprocessor speaker_a.wav podcast.otio"
            " --backend sensevoice\n"
            "\n"
            "  SenseVoice + VAD（無音区間を推論から除外）:\n"
            "    podcast-preprocessor speaker_a.wav podcast.otio"
            " --backend sensevoice --sensevoice-vad\n"
            "\n"
            "出力OTIOと同じ場所に、字幕ファイル（.srt）も生成されます。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="入力音声ファイル（話者ごとに1ファイル。指定順に "
        "source_1, source_2, ... になります）",
    )
    parser.add_argument("output", help="出力するOTIOファイルのパス")

    common = parser.add_argument_group("共通のオプション")
    common.add_argument(
        "--backend",
        choices=("faster-whisper", "sensevoice"),
        default="faster-whisper",
        help="文字起こしのバックエンド（既定: faster-whisper）",
    )
    common.add_argument(
        "--srt-output",
        help="字幕SRTの出力先（既定: 出力OTIOと同じ場所・"
        "同じ名前で拡張子を .srt にしたもの）",
    )
    common.add_argument(
        "--fps", type=int, default=30, help="タイムラインのフレームレート（既定: 30）"
    )
    common.add_argument(
        "--sequence-name",
        default="Podcast_Timeline",
        help="OTIO内のシーケンス名（既定: Podcast_Timeline）",
    )
    common.add_argument(
        "--acoustic-fillers",
        action="store_true",
        help="音声信号からもフィラー候補を検出し、音声クリップに黄色マーカーで付ける"
        "（SenseVoiceのようにフィラーを文字起こしへ残さないバックエンドでも候補を拾えます）",
    )

    whisper = parser.add_argument_group("faster-whisper のオプション")
    whisper.add_argument(
        "--model",
        default="small",
        help="Whisperモデルサイズ（tiny / base / small / medium / "
        "large-v3 / turbo など。既定: small）",
    )
    whisper.add_argument(
        "--device",
        default="auto",
        help="推論デバイス（auto / cpu / cuda。既定: auto）",
    )
    whisper.add_argument(
        "--compute-type",
        default="default",
        help="計算型（default / int8 / float16 など。"
        "CPUでは int8 が軽量です。既定: default）",
    )
    whisper.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU推論のスレッド数（0 = 自動、既定: 0）",
    )

    sensevoice = parser.add_argument_group("SenseVoice のオプション")
    sensevoice.add_argument(
        "--sensevoice-model",
        default=None,
        help="ローカルのSenseVoiceモデルディレクトリ（省略時は初回実行時にキャッシュへダウンロード）",
    )
    sensevoice.add_argument(
        "--sensevoice-quantization",
        choices=("none", "int8", "int4"),
        default="int8",
        help="ONNXモデルの量子化（既定: int8。"
        "int4 は公式配布モデルが登録されている場合のみ）",
    )
    sensevoice.add_argument(
        "--sensevoice-cache-dir",
        help="SenseVoiceモデルのキャッシュ先（既定: ~/.cache/podcastpreprocessor/）",
    )
    sensevoice.add_argument(
        "--sensevoice-audio-cache-dir",
        help="16kHzモノラルWAVへの変換キャッシュ先"
        "（既定: ~/.cache/podcastpreprocessor/converted）",
    )
    sensevoice.add_argument(
        "--sensevoice-num-threads",
        type=int,
        default=4,
        help="CPU推論のスレッド数（既定: 4）",
    )
    sensevoice.add_argument(
        "--sensevoice-provider",
        choices=("cpu", "cuda", "coreml"),
        default="cpu",
        help="ONNX Runtimeのプロバイダ（既定: cpu）",
    )
    sensevoice.add_argument(
        "--sensevoice-chunk-duration-s",
        type=float,
        default=30.0,
        help="文字起こしを進めるチャンクの長さ（秒、既定: 30）。"
        "途中経過はこの単位で保存され、再実行時に再開できます",
    )
    sensevoice.add_argument(
        "--batch-size-s",
        type=int,
        default=60,
        help="1回の推論にまとめる音声の長さ（秒、既定: 60）",
    )
    sensevoice.add_argument(
        "--language", default="ja", help="SenseVoiceの言語コード（既定: ja）"
    )
    sensevoice.add_argument(
        "--use-itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="逆テキスト正規化（ITN: 数字・日付などの表記変換）を使う"
        "（既定: 有効。無効化は --no-use-itn）",
    )
    sensevoice.add_argument(
        "--offline",
        action="store_true",
        help="モデルをダウンロードせず、キャッシュ済みのモデルだけで実行する",
    )

    vad = parser.add_argument_group(
        "SenseVoice VAD のオプション（--sensevoice-vad 指定時）"
    )
    vad.add_argument(
        "--sensevoice-vad",
        action="store_true",
        help="Silero VADで発話区間だけを検出して推論する"
        "（無音が多い音声で高速化します）",
    )
    vad.add_argument(
        "--sensevoice-vad-model",
        help="ローカルのSilero VAD ONNXモデル（省略時はキャッシュへダウンロード）",
    )
    vad.add_argument(
        "--sensevoice-vad-threshold",
        type=float,
        default=0.5,
        help="発話確率のしきい値（既定: 0.5）",
    )
    vad.add_argument(
        "--sensevoice-vad-min-silence-s",
        type=float,
        default=0.25,
        help="発話区間を終端する最小の無音長（秒、既定: 0.25）",
    )
    vad.add_argument(
        "--sensevoice-vad-min-speech-s",
        type=float,
        default=0.1,
        help="保持する最小の発話長（秒、既定: 0.1）",
    )
    vad.add_argument(
        "--sensevoice-vad-max-speech-s",
        type=float,
        default=120.0,
        help="1つの発話区間の最大長（秒、既定: 120）。長い発話だけこの長さで分割します",
    )
    vad.add_argument(
        "--sensevoice-vad-padding-s",
        type=float,
        default=0.25,
        help="発話区間の前後に付ける余白（秒、既定: 0.25）",
    )
    vad.add_argument(
        "--sensevoice-vad-threshold-db",
        type=float,
        default=-40.0,
        help="エネルギーVADのしきい値（最大フレームに対するdB、既定: -40）",
    )

    cache = parser.add_argument_group("キャッシュのオプション")
    cache.add_argument(
        "--cache-dir",
        default=None,
        help="文字起こしキャッシュの保存先"
        "（既定: ~/.cache/podcastpreprocessor/transcripts）",
    )
    cache.add_argument(
        "--no-cache",
        action="store_true",
        help="文字起こしキャッシュを使わない・保存しない",
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
            cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()

        def _build() -> Transcriber:
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
                cpu_threads=args.cpu_threads,
            )

        def _on_cache_hit(audio_path: str) -> None:
            sys.stderr.write(f"Using cached transcript: {Path(audio_path).name}\n")

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
    sources: list[tuple[AudioSource, list[RawSegment]]] = []
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
    filler_regions: dict[str, list[FillerRegion]] | None = None
    if args.acoustic_fillers:
        try:
            filler_regions = {}
            for source, _raw in sources:
                regions = detect_filler_candidates(source.path)
                filler_regions[source.source_id] = regions
                sys.stderr.write(
                    f"Acoustic filler candidates in "
                    f"{Path(source.path).name}: {len(regions)}\n"
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
