"""Download and cache official sherpa-onnx SenseVoice models."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen


@dataclass(frozen=True)
class SenseVoiceModelSpec:
    quantization: str
    archive_url: str
    archive_name: str
    model_name: str
    sha256: str | None = None


_MODEL_RELEASE = "2024-07-17"
_MODEL_SPECS = {
    "int8": SenseVoiceModelSpec(
        quantization="int8",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
        ),
        archive_name="sherpa-onnx-sense-voice-int8.tar.bz2",
        model_name="model.int8.onnx",
        sha256="7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
    ),
}


class SenseVoiceModelManager:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        timeout: float = 60.0,
    ) -> None:
        default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self._cache_dir = Path(cache_dir or default_cache / "podcastpreprocessor")
        self._timeout = timeout

    def resolve(
        self,
        quantization: str = "int8",
        model_dir: str | Path | None = None,
        offline: bool = False,
    ) -> tuple[Path, Path]:
        if model_dir is not None:
            return self._validate_model_dir(Path(model_dir), quantization)
        try:
            spec = _MODEL_SPECS[quantization]
        except KeyError as error:
            supported = ", ".join(sorted(_MODEL_SPECS))
            raise RuntimeError(
                f"SenseVoiceの量子化モデル {quantization!r} は未提供です。"
                f"利用可能: {supported}"
            ) from error

        target_dir = self._cache_dir / "sensevoice" / _MODEL_RELEASE / quantization
        model_path = target_dir / spec.model_name
        tokens_path = target_dir / "tokens.txt"
        if model_path.is_file() and tokens_path.is_file():
            return model_path, tokens_path
        if offline:
            raise RuntimeError(
                f"SenseVoiceモデルがキャッシュにありません: {target_dir}."
                " --offlineを外して初回ダウンロードを実行してください。"
            )
        self._download_and_extract(spec, target_dir)
        return self._validate_model_dir(target_dir, quantization)

    def _download_and_extract(
        self, spec: SenseVoiceModelSpec, target_dir: Path
    ) -> None:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="sensevoice-", dir=target_dir.parent) as temp:
            temp_dir = Path(temp)
            archive_path = temp_dir / spec.archive_name
            try:
                with urlopen(spec.archive_url, timeout=self._timeout) as response:
                    with archive_path.open("wb") as output:
                        shutil.copyfileobj(response, output)
            except OSError as error:
                raise RuntimeError(
                    f"SenseVoiceモデルのダウンロードに失敗しました: {spec.archive_url}"
                ) from error
            if spec.sha256 is not None:
                digest = _sha256(archive_path)
                if digest != spec.sha256:
                    raise RuntimeError(
                        "SenseVoiceモデルのSHA256検証に失敗しました。"
                        f" expected={spec.sha256} actual={digest}"
                    )
            extract_dir = temp_dir / "extract"
            extract_dir.mkdir()
            _safe_extract(archive_path, extract_dir)
            model_file = next(extract_dir.rglob(spec.model_name), None)
            tokens_file = next(extract_dir.rglob("tokens.txt"), None)
            if model_file is None or tokens_file is None:
                raise RuntimeError(
                    "ダウンロードしたSenseVoiceモデルに model または tokens.txt がありません"
                )
            if target_dir.exists():
                shutil.rmtree(target_dir)
            staging_dir = target_dir.with_name(f"{target_dir.name}.tmp")
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir()
            shutil.copy2(model_file, staging_dir / spec.model_name)
            shutil.copy2(tokens_file, staging_dir / "tokens.txt")
            os.replace(staging_dir, target_dir)

    def _validate_model_dir(
        self, model_dir: Path, quantization: str
    ) -> tuple[Path, Path]:
        model_name = (
            "model.onnx" if quantization == "none" else f"model.{quantization}.onnx"
        )
        model_path = model_dir / model_name
        tokens_path = model_dir / "tokens.txt"
        if not model_path.is_file() or not tokens_path.is_file():
            raise RuntimeError(
                f"SenseVoiceモデルディレクトリに {model_name} と tokens.txt が必要です: {model_dir}"
            )
        return model_path, tokens_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination.resolve() not in member_path.parents:
                raise RuntimeError("モデルアーカイブに不正なパスが含まれています")
        archive.extractall(destination)
