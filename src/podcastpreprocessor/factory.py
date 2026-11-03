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
    sensevoice_vad_model: str | None = None,
    sensevoice_vad_threshold: float = 0.5,
    sensevoice_vad_min_silence_s: float = 0.25,
    sensevoice_vad_min_speech_s: float = 0.1,
    sensevoice_vad_max_speech_s: float = 120.0,
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
            vad_model=sensevoice_vad_model,
            vad_threshold=sensevoice_vad_threshold,
            vad_min_silence_duration=sensevoice_vad_min_silence_s,
            vad_min_speech_duration=sensevoice_vad_min_speech_s,
            vad_max_speech_duration=sensevoice_vad_max_speech_s,
        )
    raise ValueError(f"unsupported transcription backend: {backend}")
