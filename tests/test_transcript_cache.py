from pathlib import Path

from podcastpreprocessor.models import RawSegment, WordTimestamp
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
        self, audio_paths, progress_callback=None, phase_callback=None
    ):
        self.calls.extend(audio_paths)
        return [
            (f"source_{i}", path, [_segment()], 30.0)
            for i, path in enumerate(audio_paths, start=1)
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
        build, [str(first), str(second)], SETTINGS, tmp_path
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
        [str(first), str(second)],
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
    results = transcribe_sources_cached(lambda: fake, [str(audio)], SETTINGS, None)
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
