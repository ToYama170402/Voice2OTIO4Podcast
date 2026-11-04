import opentimelineio as otio

from podcastpreprocessor.models import AudioSource, ProcessedSegment
from podcastpreprocessor.otio import generate_otio, write_otio


def test_generate_otio_builds_tracks_links_and_audio_reference(tmp_path) -> None:
    segments = [
        ProcessedSegment(
            "s1",
            1,
            0,
            30,
            0,
            30,
            "通常区間",
            source_id="speaker_a",
        ),
        ProcessedSegment(
            "s2",
            1,
            45,
            75,
            45,
            75,
            "えーと候補",
            is_cut_candidate=True,
            flag_label="[要カット: フィラー]",
            marker_reason="判定: フィラー (えーと)",
            color_label="Red",
            source_id="speaker_a",
        ),
    ]
    source = AudioSource("speaker_a", "C:/audio/speaker_a.wav", 1, 90)

    timeline = generate_otio(segments, [source], fps=30)

    assert timeline.name == "Podcast_Timeline"
    assert [track.kind for track in timeline.tracks] == [
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Video,
        otio.schema.TrackKind.Audio,
    ]
    assert len(timeline.tracks[0]) == 2
    assert len(timeline.tracks[1]) == 3
    assert len(timeline.tracks[2]) == 4

    candidate_id = timeline.tracks[0][1].metadata["Resolve_OTIO"]["Link Group ID"]
    assert (
        candidate_id == timeline.tracks[1][2].metadata["Resolve_OTIO"]["Link Group ID"]
    )
    assert (
        timeline.tracks[2][2].metadata["Resolve_OTIO"]["Link Group ID"] == candidate_id
    )
    title_clip = timeline.tracks[1][2]
    assert isinstance(title_clip.media_reference, otio.schema.GeneratorReference)
    assert title_clip.media_reference.generator_kind == "Text"
    assert title_clip.media_reference.parameters["text"] == "えーと候補"
    assert title_clip.media_reference.available_range == otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, 30),
        duration=otio.opentime.RationalTime(90, 30),
    )
    assert title_clip.source_range.start_time.value == 0
    assert isinstance(
        timeline.tracks[0][1].media_reference, otio.schema.GeneratorReference
    )
    assert timeline.tracks[0][1].media_reference.available_range == (
        title_clip.media_reference.available_range
    )
    assert timeline.tracks[0][1].metadata["marker_reason"] == "判定: フィラー (えーと)"
    assert timeline.tracks[0][1].markers[0].name == "カット候補"
    assert timeline.tracks[2][0].media_reference.target_url == "C:/audio/speaker_a.wav"

    output_path = tmp_path / "timeline.otio"
    write_otio(str(output_path), segments, [source])
    reloaded = otio.adapters.read_from_file(str(output_path))
    assert reloaded.name == "Podcast_Timeline"
