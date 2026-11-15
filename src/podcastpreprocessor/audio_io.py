"""Audio preparation layer: normalize inputs to 16 kHz mono WAV.

Transcription backends work best on 16 kHz mono audio, but inputs arrive
as anything (48 kHz stereo WAV, m4a/AAC, ...). SenseVoice used to read the
whole file with soundfile and resample in memory with numpy
interpolation: slow, single-threaded, and several full-size copies in
RAM. This layer converts each input once, streaming through PyAV's
resampler (libswresample), and caches the converted file so later runs
skip the work. As a side effect, formats soundfile cannot read at all
(such as m4a) become usable, because PyAV decodes them.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

TARGET_SAMPLE_RATE = 16000


def default_converted_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "podcastpreprocessor" / "converted"


def _file_key(audio_path: str) -> str:
    stat = Path(audio_path).stat()
    payload = {
        "path": str(Path(audio_path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "target": TARGET_SAMPLE_RATE,
    }
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
    except Exception:
        return True
    return not (info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1)


def convert_to_wav16k(audio_path: str, dest_path: Path) -> None:
    """Stream-decode and resample any audio file to a 16 kHz mono WAV."""
    try:
        import av
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "audio conversion requires PyAV and soundfile; "
            "install project dependencies first"
        ) from error
    container = av.open(audio_path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise RuntimeError(f"no audio stream found: {audio_path}")
    resampler = av.AudioResampler(
        format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE
    )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest_path.with_suffix(".wav.tmp")
    buffer: list[Any] = []
    buffered_samples = 0
    with sf.SoundFile(
        str(temporary),
        "w",
        samplerate=TARGET_SAMPLE_RATE,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    ) as out:

        def flush() -> None:
            nonlocal buffered_samples
            if not buffer:
                return
            import numpy as np

            block = np.concatenate(buffer)
            buffer.clear()
            buffered_samples = 0
            out.write(block)

        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                block = resampled.to_ndarray().reshape(-1)
                buffer.append(block)
                buffered_samples += len(block)
                if buffered_samples >= TARGET_SAMPLE_RATE:
                    flush()
        flush()
    container.close()
    os.replace(temporary, dest_path)


def prepare_audio(
    audio_path: str,
    cache_dir: Path | None = None,
    soundfile_module: Any = None,
) -> str:
    """Return a path to a 16 kHz mono WAV version of the input.

    Files that are already readable 16 kHz mono audio are returned
    unchanged. Anything else is converted once and cached under
    ``cache_dir`` (default: ~/.cache/podcastpreprocessor/converted).
    """
    if soundfile_module is None:
        try:
            import soundfile as soundfile_module
        except ImportError:
            return audio_path
    if not _needs_conversion(audio_path, soundfile_module):
        return audio_path
    directory = cache_dir if cache_dir is not None else default_converted_dir()
    dest = directory / f"{_file_key(audio_path)}.wav"
    if not dest.exists():
        convert_to_wav16k(audio_path, dest)
    return str(dest)
