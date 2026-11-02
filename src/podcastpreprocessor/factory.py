"""Factories for selecting a transcription backend."""

from .sensevoice import SenseVoiceTranscriber
from .transcriber import FasterWhisperTranscriber, Transcriber


def build_transcriber(
    backend: str,
    model: str,
    device: str,
    compute_type: str,
    sensevoice_model: str,
    language: str,
    use_itn: bool,
    batch_size_s: int,
    sensevoice_quantization: str = "int8",
    sensevoice_model_dir: str | None = None,
    sensevoice_cache_dir: str | None = None,
    sensevoice_offline: bool = False,
    sensevoice_provider: str = "cpu",
    sensevoice_num_threads: int = 4,
    sensevoice_chunk_duration_s: float = 30.0,
    sensevoice_vad: bool = False,
    sensevoice_vad_threshold_db: float = -40.0,
    sensevoice_vad_padding_s: float = 0.25,
) -> Transcriber:
    if backend == "faster-whisper":
        return FasterWhisperTranscriber(model, device, compute_type)
    if backend == "sensevoice":
        return SenseVoiceTranscriber(
            quantization=sensevoice_quantization,
            model_dir=sensevoice_model_dir or sensevoice_model,
            cache_dir=sensevoice_cache_dir,
            offline=sensevoice_offline,
            provider=sensevoice_provider,
            num_threads=sensevoice_num_threads,
            language=language,
            use_itn=use_itn,
            chunk_duration_s=sensevoice_chunk_duration_s,
            vad=sensevoice_vad,
            vad_threshold_db=sensevoice_vad_threshold_db,
            vad_padding_s=sensevoice_vad_padding_s,
        )
    raise ValueError(f"unsupported transcription backend: {backend}")
