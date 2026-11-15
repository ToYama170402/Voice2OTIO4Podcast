import opentimelineio as otio

from podcastpreprocessor.models import AudioSource, RawSegment, WordTimestamp
from podcastpreprocessor.pipeline import process_sources


def test_process_sources_writes_one_three_layer_group_per_source(tmp_path) -> None:
    source_a = AudioSource("source_a", "C:/audio/a.wav", 1, 90)
    source_b = AudioSource("source_b", "C:/audio/b.wav", 2, 60)
    raw_a = [
        RawSegment(
            1,
            0.0,
            1.0,
            "Aです。",
            0.0,
            [WordTimestamp("Aです。", 0.0, 1.0, 1.0)],
            source_id="source_a",
        )
    ]
    raw_b = [
        RawSegment(
            2,
            0.5,
            1.5,
            "Bです。",
            0.0,
            [WordTimestamp("Bです。", 0.5, 1.5, 1.0)],
            source_id="source_b",
        )
    ]
    output_path = tmp_path / "timeline.otio"

    processed = process_sources(
        [(source_a, raw_a), (source_b, raw_b)], str(output_path)
    )

    assert {segment.source_id for segment in processed} == {"source_a", "source_b"}
    timeline = otio.adapters.read_from_file(str(output_path))
    assert len(timeline.tracks) == 6
    assert [track.kind for track in timeline.tracks] == [
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Audio,
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Audio,
    ]
    assert (tmp_path / "timeline.srt").read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "Aです。\n"
        "\n"
        "2\n"
        "00:00:00,500 --> 00:00:01,500\n"
        "Bです。\n"
        "\n"
    )


def test_process_sources_marks_acoustic_filler_candidates_on_audio_clips(
    tmp_path,
) -> None:
    from podcastpreprocessor.filler_detector import FillerRegion

    source = AudioSource("source_1", "C:/audio/a.wav", 1, 90)
    raw = [
        RawSegment(
            1,
            0.0,
            3.0,
            "今日は収録です",
            0.0,
            [WordTimestamp("今日は収録です", 0.0, 3.0, 1.0)],
            source_id="source_1",
        )
    ]
    output_path = tmp_path / "timeline.otio"

    processed = process_sources(
        [(source, raw)],
        str(output_path),
        filler_regions={"source_1": [FillerRegion(1.0, 1.6)]},
    )

    assert len(processed[0].clip_markers) == 1
    marker = processed[0].clip_markers[0]
    assert marker.name == "フィラー候補"
    assert (marker.start_frame, marker.end_frame) == (30, 48)

    timeline = otio.adapters.read_from_file(str(output_path))
    audio_track = next(t for t in timeline.tracks if t.kind == "Audio")
    clip = next(item for item in audio_track if hasattr(item, "markers"))
    assert len(clip.markers) == 1
    assert clip.markers[0].name == "フィラー候補"
    marked = clip.markers[0].marked_range
    assert marked.start_time.value == 30
    assert marked.duration.value == 18


def test_process_sources_mirrors_filler_markers_onto_timeline(tmp_path) -> None:
    """Timeline (stack) markers are the form Resolve reliably imports."""
    from podcastpreprocessor.filler_detector import FillerRegion

    source = AudioSource("source_1", "C:/audio/a.wav", 1, 90)
    raw = [
        RawSegment(
            1,
            0.0,
            3.0,
            "今日は収録です",
            0.0,
            [WordTimestamp("今日は収録です", 0.0, 3.0, 1.0)],
            source_id="source_1",
        )
    ]
    output_path = tmp_path / "timeline.otio"

    process_sources(
        [(source, raw)],
        str(output_path),
        filler_regions={"source_1": [FillerRegion(1.0, 1.6)]},
    )

    timeline = otio.adapters.read_from_file(str(output_path))
    stack_markers = list(timeline.tracks.markers)
    assert len(stack_markers) == 1
    assert stack_markers[0].name == "フィラー候補"
    assert stack_markers[0].color == "YELLOW"
    assert stack_markers[0].marked_range.start_time.value == 30
    assert stack_markers[0].marked_range.duration.value == 18
