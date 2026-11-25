"""On-disk cache of transcription results, so a failed run can resume.

Transcription is the expensive stage and everything after it can fail
(segmentation, OTIO generation, a crash, a power loss). Each file's raw
segments are stored as JSON right after that file is transcribed; a later
run with the same audio file and the same transcription settings loads
them instead of transcribing again.

A run can also die *during* a long transcription. For backends that
decode in independent chunks (SenseVoice), ChunkCache additionally
journals every finished chunk to an append-only JSONL file, so a rerun
only decodes the chunks that never completed. Both layers are
best-effort: any read or validation problem is treated as a miss, never
as an error.
"""

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import AudioInput, RawSegment, WordTimestamp
from .transcriber import (
    PhaseCallback,
    ProgressCallback,
    Transcriber,
    TranscriptionResult,
)

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "podcastpreprocessor" / "transcripts"


def cache_key(audio_path: str, settings: Mapping[str, object]) -> str:
    """Identify a transcription by file identity plus backend settings."""
    stat = Path(audio_path).stat()
    payload = {
        "audio": {
            "path": str(Path(audio_path).resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "settings": dict(settings),
        "version": CACHE_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _segment_to_dict(segment: RawSegment) -> dict[str, object]:
    return {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "no_speech_prob": segment.no_speech_prob,
        "words": [
            {
                "word": word.word,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in segment.words
        ],
    }


def _segment_from_dict(item: Mapping[str, Any], source_id: str) -> RawSegment:
    words = [
        WordTimestamp(
            word=str(word["word"]),
            start=float(word["start"]),
            end=float(word["end"]),
            probability=float(word["probability"]),
        )
        for word in item["words"]
    ]
    return RawSegment(
        id=int(item["id"]),
        start=float(item["start"]),
        end=float(item["end"]),
        text=str(item["text"]),
        no_speech_prob=float(item["no_speech_prob"]),
        words=words,
        source_id=source_id,
    )


def load_cached(
    cache_dir: Path, key: str, source_id: str
) -> tuple[list[RawSegment], float | None] | None:
    try:
        data = json.loads((cache_dir / f"{key}.json").read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return None
        segments = [_segment_from_dict(item, source_id) for item in data["segments"]]
        duration = data.get("duration")
        return segments, (float(duration) if duration is not None else None)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def save_cached(
    cache_dir: Path,
    key: str,
    segments: Sequence[RawSegment],
    duration: float | None,
) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "duration": duration,
            "segments": [_segment_to_dict(segment) for segment in segments],
        }
        target = cache_dir / f"{key}.json"
        temporary = cache_dir / f"{key}.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
    except OSError:
        # A cache that cannot be written only costs a re-transcription.
        pass


def transcribe_sources_cached(
    build_transcriber: Callable[[], Transcriber],
    audio_inputs: Sequence[AudioInput],
    settings: Mapping[str, object],
    cache_dir: Path | None,
    progress_callback: ProgressCallback | None = None,
    phase_callback: PhaseCallback | None = None,
    on_cache_hit: Callable[[str], None] | None = None,
) -> list[TranscriptionResult]:
    """Transcribe each input, reusing cached results whenever possible.

    The transcriber is built lazily on the first cache miss, so a fully
    cached run never loads a model. Each input is saved to the cache as
    soon as its transcription finishes, before any later stage runs.
    """
    results: list[TranscriptionResult] = []
    transcriber: Transcriber | None = None
    for index, spec in enumerate(audio_inputs, start=1):
        audio_path = spec.path
        source_id = f"source_{index}"
        key: str | None = None
        if cache_dir is not None:
            try:
                key_settings: Mapping[str, object] = settings
                if spec.stream_index:
                    # Stream 0 keeps the historical key so caches stay valid.
                    key_settings = {**settings, "stream_index": spec.stream_index}
                key = cache_key(audio_path, key_settings)
            except OSError:
                key = None
        if key is not None and cache_dir is not None:
            cached = load_cached(cache_dir, key, source_id)
            if cached is not None:
                segments, duration = cached
                if on_cache_hit is not None:
                    on_cache_hit(audio_path)
                if progress_callback is not None:
                    progress_callback(audio_path, duration or 0.0, duration)
                results.append((source_id, audio_path, segments, duration))
                continue
        if transcriber is None:
            transcriber = build_transcriber()
        transcribed = transcriber.transcribe_sources(
            [spec],
            progress_callback=progress_callback,
            phase_callback=phase_callback,
        )
        _, _, segments, duration = transcribed[0]
        for segment in segments:
            segment.source_id = source_id
        if key is not None and cache_dir is not None:
            save_cached(cache_dir, key, segments, duration)
        results.append((source_id, audio_path, segments, duration))
    return results


def _parse_journal_line(line: str) -> tuple[int, RawSegment | None] | None:
    """Parse one chunk-journal line; None means the line is unusable."""
    try:
        item = json.loads(line)
        segment = item["segment"]
        return int(item["chunk"]), (
            _segment_from_dict(segment, "source_1") if segment is not None else None
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


class ChunkCache:
    """Append-only journal of finished chunks for one transcription.

    Each completed chunk is appended (and fsynced) as one JSON line, so a
    crash or kill loses at most the chunk that was being decoded. A torn
    final line is ignored on load.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[int, RawSegment | None]:
        entries: dict[int, RawSegment | None] = {}
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return entries
        for line in lines:
            parsed = _parse_journal_line(line)
            if parsed is not None:
                entries[parsed[0]] = parsed[1]
        return entries

    def store(self, index: int, segment: RawSegment | None) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {
                    "chunk": index,
                    "segment": _segment_to_dict(segment) if segment else None,
                },
                ensure_ascii=False,
            )
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass

    def clear(self) -> None:
        with contextlib.suppress(OSError):
            self._path.unlink()
