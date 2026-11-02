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
