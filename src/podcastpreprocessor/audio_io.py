"""Audio preparation layer: normalize inputs to 16 kHz mono WAV.

Transcription backends work best on 16 kHz mono audio, but inputs arrive
as anything (48 kHz stereo WAV, m4a/AAC, ...). SenseVoice used to read the
whole file with soundfile and resample in memory with numpy
interpolation: slow, single-threaded, and several full-size copies in
RAM. This layer converts each input once, streaming through PyAV's
resampler (libswresample), and caches the converted file so later runs
skip the work. A conversion that is interrupted mid-file resumes from a
journaled partial instead of starting over. As a side effect, formats
soundfile cannot read at all (such as m4a) become usable, because PyAV
decodes them.
"""

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import IO, Any

from .models import AudioStreamInfo

TARGET_SAMPLE_RATE = 16000


def list_audio_streams(audio_path: str) -> list[AudioStreamInfo]:
    """List the audio streams (tracks) inside a media file, in order."""
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "audio stream probing requires PyAV; install project dependencies first"
        ) from error
    container = av.open(audio_path)
    try:
        return [
            AudioStreamInfo(
                stream_index=index,
                channels=int(stream.codec_context.channels or 0),
                sample_rate=int(stream.codec_context.sample_rate or 0),
            )
            for index, stream in enumerate(container.streams.audio)
        ]
    finally:
        container.close()


def default_converted_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "podcastpreprocessor" / "converted"


def _file_key(audio_path: str, stream_index: int = 0) -> str:
    stat = Path(audio_path).stat()
    payload: dict[str, object] = {
        "path": str(Path(audio_path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "target": TARGET_SAMPLE_RATE,
    }
    if stream_index:
        # Stream 0 keeps the historical key so existing caches stay valid.
        payload["stream"] = stream_index
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _needs_conversion(audio_path: str, soundfile_module: Any) -> bool:
    """True unless the file is already a readable 16 kHz mono WAV-like file."""
    info_fn = getattr(soundfile_module, "info", None)
    if info_fn is None:
        # A minimal soundfile stand-in (e.g. in tests) cannot probe; leave
        # the file untouched and let the caller's read path handle it.
        return False
    try:
        info = info_fn(audio_path)
    except Exception:  # noqa: BLE001 -- 読めないファイルは「変換が必要」として扱う
        return True
    return not (info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1)


def _partial_paths(dest_path: Path) -> tuple[Path, Path]:
    """(raw PCM partial, journal) paths used while a conversion runs."""
    return (
        dest_path.with_suffix(".pcm"),
        dest_path.parent / f"{dest_path.stem}.journal.json",
    )


def _load_resume_samples(partial: Path, journal: Path) -> int:
    """Samples safely stored by an interrupted conversion (0 = start over).

    The journal is only advanced after the PCM bytes are fsynced, so a
    journal entry never claims more than what is on disk. A partial file
    that disagrees with the journal is truncated to the journal, or the
    conversion restarts when even that is impossible.
    """
    try:
        samples = int(json.loads(journal.read_text(encoding="utf-8"))["samples"])
        size = partial.stat().st_size
    except (OSError, ValueError, KeyError, TypeError):
        return 0
    if samples <= 0:
        return 0
    expected = samples * 2  # int16 mono
    if size < expected:
        return 0
    if size > expected:
        try:
            with partial.open("r+b") as handle:
                handle.truncate(expected)
        except OSError:
            return 0
    return samples


def _save_journal(journal: Path, samples: int) -> None:
    try:
        temporary = journal.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"version": 1, "samples": samples}), encoding="utf-8"
        )
        temporary.replace(journal)
    except OSError:
        pass


def _wrap_pcm_as_wav(partial: Path, dest_path: Path, sf: Any) -> None:
    """Wrap the raw 16 kHz mono int16 PCM partial into the final WAV."""
    import numpy as np

    temporary = dest_path.with_suffix(".wav.tmp")
    with (
        partial.open("rb") as source,
        sf.SoundFile(
            str(temporary),
            "w",
            samplerate=TARGET_SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",
            format="WAV",
        ) as out,
    ):
        while block := source.read(1 << 20):
            out.write(np.frombuffer(block, dtype=np.int16))
    temporary.replace(dest_path)


def _audio_stream_or_raise(
    container: Any, audio_path: str, stream_index: int = 0
) -> Any:
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        raise RuntimeError(f"no audio stream found: {audio_path}")
    if not 0 <= stream_index < len(audio_streams):
        container.close()
        raise RuntimeError(
            f"audio stream {stream_index} not found in {audio_path}: "
            f"the file has {len(audio_streams)} audio stream(s)"
        )
    return audio_streams[stream_index]


def _seek_to_resume(container: Any, stream: Any, resume_time: float) -> bool:
    """Seek near the resume point; False when the container cannot seek."""
    try:
        # With a stream given, seek timestamps are in that stream's
        # time_base units (not AV_TIME_BASE microseconds).
        seek_ts = (
            int(resume_time / float(stream.time_base))
            if stream.time_base
            else int(resume_time * 1_000_000)
        )
        container.seek(seek_ts, stream=stream, backward=True)
    except Exception:  # noqa: BLE001 -- seekできない場合は最初から変換し直す
        return False
    return True


class _ResampleSink:
    """Appends resampled PCM to the partial file, journaling progress.

    ``written`` counts the samples safely attributable to this conversion,
    including the ones a previous interrupted run left behind. The first
    decoded frame fixes how many resampled samples to drop at the seam so
    the resumed output lines up with the journal.
    """

    def __init__(self, journal: Path, written: int, resume_time: float) -> None:
        self._journal = journal
        self._resume_time = resume_time
        self.written = written
        self._since_journal = 0
        self._skip: int | None = None

    def run(self, partial: Path, container: Any, stream: Any, resampler: Any) -> None:
        with partial.open("ab" if self.written else "wb") as out:
            for frame in container.decode(stream):
                self._consume_frame(out, frame, resampler)
            self._drain(out, resampler)
            out.flush()
            os.fsync(out.fileno())

    def _consume_frame(self, out: IO[bytes], frame: Any, resampler: Any) -> None:
        if self._skip is None:
            start = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None and frame.time_base
                else self._resume_time
            )
            self._skip = max(0, round((self._resume_time - start) * TARGET_SAMPLE_RATE))
        for resampled in resampler.resample(frame):
            self._append_resampled(out, resampled)

    def _drain(self, out: IO[bytes], resampler: Any) -> None:
        for resampled in resampler.resample(None):
            self._append_resampled(out, resampled)

    def _append_resampled(self, out: IO[bytes], resampled: Any) -> None:
        block = resampled.to_ndarray().reshape(-1)
        if self._skip:
            drop = min(self._skip, len(block))
            block = block[drop:]
            self._skip -= drop
        if len(block):
            self._append(out, block)

    def _append(self, out: IO[bytes], block: Any) -> None:
        out.write(block.tobytes())
        self.written += len(block)
        self._since_journal += len(block)
        # Journal (and fsync) in 30-second strides: conversion is
        # fast, so losing up to 30 s of it on a crash is cheap, and
        # per-second fsyncs would dominate on slow disks.
        if self._since_journal >= TARGET_SAMPLE_RATE * 30:
            out.flush()
            os.fsync(out.fileno())
            _save_journal(self._journal, self.written)
            self._since_journal = 0


def convert_to_wav16k(audio_path: str, dest_path: Path, stream_index: int = 0) -> None:
    """Stream-decode and resample any audio file to a 16 kHz mono WAV.

    The conversion is resumable: resampled PCM is appended to a partial
    file and a journal records how many samples are safely stored, so an
    interrupted conversion continues near where it stopped instead of
    starting over. The journal is best-effort; any inconsistency simply
    restarts the conversion.
    """
    try:
        import av
    except ImportError as error:
        raise RuntimeError(
            "audio conversion requires PyAV and soundfile; "
            "install project dependencies first"
        ) from error
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    partial, journal = _partial_paths(dest_path)
    written = _load_resume_samples(partial, journal)
    resume_time = written / TARGET_SAMPLE_RATE

    container = av.open(audio_path)
    stream = _audio_stream_or_raise(container, audio_path, stream_index)
    if written and not _seek_to_resume(container, stream, resume_time):
        # The container cannot seek; restart the conversion cleanly.
        container.close()
        container = av.open(audio_path)
        stream = _audio_stream_or_raise(container, audio_path, stream_index)
        written = 0
        resume_time = 0.0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
    sink = _ResampleSink(journal, written, resume_time)
    sink.run(partial, container, stream, resampler)
    container.close()
    _save_journal(journal, sink.written)
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "audio conversion requires PyAV and soundfile; "
            "install project dependencies first"
        ) from error
    _wrap_pcm_as_wav(partial, dest_path, sf)
    for path in (partial, journal):
        with contextlib.suppress(OSError):
            path.unlink()


def prepare_audio(
    audio_path: str,
    cache_dir: Path | None = None,
    soundfile_module: Any = None,
    stream_index: int = 0,
) -> str:
    """Return a path to a 16 kHz mono WAV version of the input.

    Files that are already readable 16 kHz mono audio are returned
    unchanged. Anything else is converted once and cached under
    ``cache_dir`` (default: ~/.cache/podcastpreprocessor/converted).
    A ``stream_index`` above 0 always converts: the fast path can only
    ever describe the file's first (or only) audio stream.
    """
    if stream_index == 0:
        if soundfile_module is None:
            try:
                import soundfile
            except ImportError:
                return audio_path
            soundfile_module = soundfile
        if not _needs_conversion(audio_path, soundfile_module):
            return audio_path
    directory = cache_dir if cache_dir is not None else default_converted_dir()
    dest = directory / f"{_file_key(audio_path, stream_index)}.wav"
    if not dest.exists():
        convert_to_wav16k(audio_path, dest, stream_index)
    return str(dest)
