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
    # Resolve drops clip markers when the media reference has no
    # available_range, so the audio reference must carry the full
    # source range.
    assert timeline.tracks[2][0].media_reference.available_range == (
        otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(90, 30),
        )
    )

    output_path = tmp_path / "timeline.otio"
    write_otio(str(output_path), segments, [source])
    reloaded = otio.adapters.read_from_file(str(output_path))
    assert reloaded.name == "Podcast_Timeline"


def test_generate_otio_routes_clips_to_their_stream_of_a_multi_track_file() -> None:
    segments = [
        ProcessedSegment("s1", 2, 0, 30, 0, 30, "2番目のマイク", source_id="speaker_b")
    ]
    source = AudioSource(
        "speaker_b",
        "C:/audio/output.mkv",
        2,
        30,
        stream_index=1,
        stream_count=2,
        channels=2,
    )

    timeline = generate_otio(segments, [source], fps=30)

    audio_track = timeline.tracks[2]
    assert audio_track.name == "speaker_b 音声トラック"
    routed = [
        [dict(entry) for entry in clip.metadata["Resolve_OTIO"]["Channels"]]
        for clip in audio_track
        if isinstance(clip, otio.schema.Clip)
    ]
    assert routed
    assert all(
        channels
        == [
            {"Source Channel ID": 0, "Source Track ID": 1},
            {"Source Channel ID": 1, "Source Track ID": 1},
        ]
        for channels in routed
    )


def test_generate_otio_omits_channels_for_single_stream_files() -> None:
    segments = [ProcessedSegment("s1", 1, 0, 30, 0, 30, "通常", source_id="speaker_a")]
    source = AudioSource("speaker_a", "C:/audio/speaker_a.wav", 1, 30)

    timeline = generate_otio(segments, [source], fps=30)

    for clip in timeline.tracks[2]:
        if isinstance(clip, otio.schema.Clip):
            assert "Channels" not in clip.metadata.get("Resolve_OTIO", {})
