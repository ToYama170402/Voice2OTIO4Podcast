from io import BytesIO
from pathlib import Path
import tarfile
from dataclasses import replace

import pytest

import podcastpreprocessor.model_manager as model_manager
from podcastpreprocessor.model_manager import SenseVoiceModelManager


def _archive_bytes(model_name: str = "model.int8.onnx") -> bytes:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, content in (
            (f"sensevoice/{model_name}", b"onnx"),
            ("sensevoice/tokens.txt", b"a 1\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return output.getvalue()


def test_model_manager_downloads_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _archive_bytes()
    calls: list[str] = []

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(url: str, timeout: float) -> Response:
        calls.append(url)
        return Response(payload)

    monkeypatch.setitem(
        model_manager._MODEL_SPECS,
        "int8",
        replace(model_manager._MODEL_SPECS["int8"], sha256=None),
    )
    monkeypatch.setattr("podcastpreprocessor.model_manager.urlopen", fake_urlopen)
    manager = SenseVoiceModelManager(tmp_path)

    first = manager.resolve()
    second = manager.resolve()

    assert first == second
    assert first[0].read_bytes() == b"onnx"
    assert len(calls) == 1


def test_model_manager_offline_requires_cached_model(tmp_path: Path) -> None:
    manager = SenseVoiceModelManager(tmp_path)

    with pytest.raises(RuntimeError, match="キャッシュにありません"):
        manager.resolve(offline=True)


def test_model_manager_downloads_and_reuses_vad_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"silero"
    calls: list[str] = []

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(url: str, timeout: float) -> Response:
        calls.append(url)
        return Response(payload)

    monkeypatch.setattr("podcastpreprocessor.model_manager.urlopen", fake_urlopen)
    monkeypatch.setattr(
        model_manager,
        "_VAD_MODEL_SPEC",
        replace(model_manager._VAD_MODEL_SPEC, sha256=None),
    )
    manager = SenseVoiceModelManager(tmp_path)

    first = manager.resolve_vad()
    second = manager.resolve_vad()

    assert first == second
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_model_manager_vad_offline_requires_cached_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="キャッシュにありません"):
        SenseVoiceModelManager(tmp_path).resolve_vad(offline=True)


def test_model_manager_vad_accepts_explicit_model(tmp_path: Path) -> None:
    model_path = tmp_path / "silero_vad.onnx"
    model_path.write_bytes(b"silero")

    assert SenseVoiceModelManager(tmp_path).resolve_vad(model_path) == model_path


def test_model_manager_rejects_archive_path_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, BytesIO(b"x"))

    class Response(BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        "podcastpreprocessor.model_manager.urlopen",
        lambda *args, **kwargs: Response(output.getvalue()),
    )
    monkeypatch.setitem(
        model_manager._MODEL_SPECS,
        "int8",
        replace(model_manager._MODEL_SPECS["int8"], sha256=None),
    )

    with pytest.raises(RuntimeError, match="不正なパス"):
        SenseVoiceModelManager(tmp_path).resolve()
