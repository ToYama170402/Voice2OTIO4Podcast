from pathlib import Path

import pytest

from podcastpreprocessor.models import AudioInput, RawSegment, WordTimestamp
from podcastpreprocessor.transcript_cache import (
    cache_key,
    load_cached,
    save_cached,
    transcribe_sources_cached,
)

SETTINGS = {"backend": "sensevoice", "model": "small"}


def _segment(source_id: str = "source_1") -> RawSegment:
    return RawSegment(
        id=3,
        start=1.0,
        end=2.5,
        text="えーと、今日は",
        no_speech_prob=0.25,
        words=[
            WordTimestamp("えーと、", 1.0, 1.8, 0.9),
            WordTimestamp("今日は", 1.8, 2.5, 0.8),
        ],
        source_id=source_id,
    )


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "episode.wav"
    path.write_bytes(b"fake audio bytes")
    return path


def test_cache_round_trip(tmp_path: Path) -> None:
    key = cache_key(str(_audio(tmp_path)), SETTINGS)
    save_cached(tmp_path, key, [_segment()], 12.5)

    loaded = load_cached(tmp_path, key, "source_2")

    assert loaded is not None
    segments, duration = loaded
    assert duration == 12.5
    assert segments[0].text == "えーと、今日は"
    assert segments[0].words[1].word == "今日は"
    # Source ids are assigned by position at load time, not stored.
    assert segments[0].source_id == "source_2"


def test_cache_key_changes_with_settings_and_file(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    base = cache_key(str(audio), SETTINGS)
    assert base != cache_key(str(audio), {"backend": "faster-whisper"})
    audio.write_bytes(b"different bytes, different size")
    assert base != cache_key(str(audio), SETTINGS)


def test_load_cached_tolerates_corruption(tmp_path: Path) -> None:
    key = cache_key(str(_audio(tmp_path)), SETTINGS)
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")
    assert load_cached(tmp_path, key, "source_1") is None
    (tmp_path / f"{key}.json").write_text('{"version": 999}', encoding="utf-8")
    assert load_cached(tmp_path, key, "source_1") is None


class _FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe_sources(
        self, audio_inputs, progress_callback=None, phase_callback=None
    ):
        self.calls.extend(spec.path for spec in audio_inputs)
        return [
            (f"source_{i}", spec.path, [_segment()], 30.0)
            for i, spec in enumerate(audio_inputs, start=1)
        ]


def test_transcribe_sources_cached_resumes_after_failure(tmp_path: Path) -> None:
    first = _audio(tmp_path)
    second = tmp_path / "track2.wav"
    second.write_bytes(b"second track")
    fake = _FakeTranscriber()
    builds = []

    def build():
        builds.append(1)
        return fake

    # First run: transcribes both files and caches them.
    results = transcribe_sources_cached(
        build, [AudioInput(str(first)), AudioInput(str(second))], SETTINGS, tmp_path
    )
    assert [item[0] for item in results] == ["source_1", "source_2"]
    assert fake.calls == [str(first), str(second)]

    # Second run: everything comes from the cache; the transcriber factory
    # is never invoked, so no model is loaded.
    fake.calls.clear()
    builds.clear()
    hits: list[str] = []
    results = transcribe_sources_cached(
        build,
        [AudioInput(str(first)), AudioInput(str(second))],
        SETTINGS,
        tmp_path,
        on_cache_hit=hits.append,
    )
    assert builds == []
    assert fake.calls == []
    assert hits == [str(first), str(second)]
    assert results[1][2][0].source_id == "source_2"


def test_transcribe_sources_cached_without_cache_dir(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    fake = _FakeTranscriber()
    results = transcribe_sources_cached(
        lambda: fake, [AudioInput(str(audio))], SETTINGS, None
    )
    assert fake.calls == [str(audio)]
    assert results[0][2][0].text == "えーと、今日は"


def test_chunk_cache_journals_and_tolerates_torn_line(tmp_path: Path) -> None:
    from podcastpreprocessor.transcript_cache import ChunkCache

    store = ChunkCache(tmp_path / "partials" / "key.jsonl")
    store.store(0, _segment())
    store.store(1, None)
    with (tmp_path / "partials" / "key.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"chunk": 2, "segm')  # torn write from a crash

    entries = store.load()

    first = entries[0]
    assert first is not None
    assert first.text == "えーと、今日は"
    assert entries[1] is None
    assert 2 not in entries
    store.clear()
    assert store.load() == {}


def test_cache_key_is_stable_hex_and_order_insensitive(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    key = cache_key(str(audio), SETTINGS)

    assert len(key) == 64
    assert all(char in "0123456789abcdef" for char in key)
    assert cache_key(str(audio), SETTINGS) == key
    reordered = {"model": "small", "backend": "sensevoice"}
    assert cache_key(str(audio), reordered) == key


def test_cache_key_changes_with_mtime_only(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    base = cache_key(str(audio), SETTINGS)

    stat = audio.stat()
    audio.touch()
    import os

    os.utime(audio, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert cache_key(str(audio), SETTINGS) != base


def test_cache_key_matches_for_relative_and_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _audio(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cache_key("episode.wav", SETTINGS) == cache_key(str(audio), SETTINGS)


def test_cache_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    key = cache_key(str(_audio(tmp_path)), SETTINGS)
    save_cached(tmp_path, key, [_segment()], None)

    loaded = load_cached(tmp_path, key, "source_2")

    assert loaded is not None
    segments, duration = loaded
    assert duration is None
    assert segments == [_segment("source_2")]


def test_load_cached_missing_file_is_a_miss(tmp_path: Path) -> None:
    assert load_cached(tmp_path, "0" * 64, "source_1") is None


def test_save_cached_creates_directories_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    key = cache_key(str(_audio(tmp_path)), SETTINGS)
    nested = tmp_path / "deep" / "cache"

    save_cached(nested, key, [_segment()], 1.0)

    assert load_cached(nested, key, "source_1") is not None
    assert not (nested / f"{key}.json.tmp").exists()
    raw = (nested / f"{key}.json").read_bytes()
    assert "えーと、今日は".encode() in raw


def test_transcribe_sources_cached_miss_side_effects(tmp_path: Path) -> None:
    first = _audio(tmp_path)
    second = tmp_path / "track2.wav"
    second.write_bytes(b"second track")
    fake = _FakeTranscriber()
    hits: list[str] = []
    phases: list[object] = []

    def phase_callback(*args: object) -> None:
        phases.append(args)

    results = transcribe_sources_cached(
        lambda: fake,
        [AudioInput(str(first)), AudioInput(str(second))],
        SETTINGS,
        tmp_path,
        phase_callback=phase_callback,
        on_cache_hit=hits.append,
    )

    assert hits == []
    # Segments returned by the transcriber are renumbered by position.
    assert results[0][2][0].source_id == "source_1"
    assert results[1][2][0].source_id == "source_2"
    assert results[0][3] == 30.0


def test_transcribe_sources_cached_hit_reports_progress(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    key = cache_key(str(audio), SETTINGS)
    save_cached(tmp_path, key, [_segment()], None)
    progress: list[tuple] = []

    def build() -> _FakeTranscriber:
        raise AssertionError("transcriber must not be built on a full hit")

    results = transcribe_sources_cached(
        build,
        [AudioInput(str(audio))],
        SETTINGS,
        tmp_path,
        progress_callback=lambda *args: progress.append(args),
    )

    assert progress == [(str(audio), 0.0, None)]
    assert results[0][3] is None


def test_transcribe_sources_cached_tolerates_missing_audio(tmp_path: Path) -> None:
    missing = tmp_path / "gone.wav"
    fake = _FakeTranscriber()

    results = transcribe_sources_cached(
        lambda: fake, [AudioInput(str(missing))], SETTINGS, tmp_path
    )

    assert fake.calls == [str(missing)]
    assert results[0][2][0].text == "えーと、今日は"
    assert list(tmp_path.glob("*.json")) == []


def test_chunk_cache_later_line_wins_and_clear_is_idempotent(
    tmp_path: Path,
) -> None:
    from podcastpreprocessor.transcript_cache import ChunkCache

    store = ChunkCache(tmp_path / "key.jsonl")
    assert store.load() == {}
    store.clear()

    store.store(0, _segment())
    updated = _segment()
    updated.text = "差し替え"
    store.store(0, updated)

    entries = store.load()
    first = entries[0]
    assert first is not None
    assert first.text == "差し替え"

    import json

    line = (tmp_path / "key.jsonl").read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    assert payload["chunk"] == 0
    assert payload["segment"]["words"][0] == {
        "word": "えーと、",
        "start": 1.0,
        "end": 1.8,
        "probability": 0.9,
    }


class _PerStreamTranscriber:
    """Fake backend whose transcript text identifies the stream."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def transcribe_sources(
        self, audio_inputs, progress_callback=None, phase_callback=None
    ):
        results = []
        for index, spec in enumerate(audio_inputs, start=1):
            self.calls.append((spec.path, spec.stream_index))
            segment = _segment()
            segment.text = f"stream {spec.stream_index}"
            results.append((f"source_{index}", spec.path, [segment], 30.0))
        return results


def test_transcribe_sources_cached_separates_streams_of_one_file(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    cache_dir = tmp_path / "cache"
    fake = _PerStreamTranscriber()
    builds: list[int] = []

    def build() -> _PerStreamTranscriber:
        builds.append(1)
        return fake

    inputs = [AudioInput(str(audio), 0), AudioInput(str(audio), 1)]
    results = transcribe_sources_cached(build, inputs, SETTINGS, cache_dir)

    assert [result[2][0].text for result in results] == ["stream 0", "stream 1"]
    assert fake.calls == [(str(audio), 0), (str(audio), 1)]
    assert len(list(cache_dir.glob("*.json"))) == 2

    again = transcribe_sources_cached(build, inputs, SETTINGS, cache_dir)

    assert builds == [1]
    assert [result[2][0].text for result in again] == ["stream 0", "stream 1"]
