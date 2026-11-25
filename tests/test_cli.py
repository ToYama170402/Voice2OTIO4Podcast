import pytest

from podcastpreprocessor import factory


def test_build_transcriber_keeps_whisper_as_default_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWhisperTranscriber:
        def __init__(self, *args: object) -> None:
            self.args = args

    monkeypatch.setattr(factory, "FasterWhisperTranscriber", FakeWhisperTranscriber)
    transcriber = factory.build_transcriber(
        "faster-whisper", "small", "cpu", "int8", "ignored", "ja", True, 60
    )

    assert isinstance(transcriber, FakeWhisperTranscriber)
    assert transcriber.args == ("small", "cpu", "int8", 0)


def test_build_transcriber_passes_cpu_threads_to_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWhisperTranscriber:
        def __init__(self, *args: object) -> None:
            self.args = args

    monkeypatch.setattr(factory, "FasterWhisperTranscriber", FakeWhisperTranscriber)
    transcriber = factory.build_transcriber(
        "faster-whisper",
        "small",
        "cpu",
        "int8",
        "ignored",
        "ja",
        True,
        60,
        cpu_threads=8,
    )

    assert isinstance(transcriber, FakeWhisperTranscriber)
    assert transcriber.args == ("small", "cpu", "int8", 8)


def test_build_transcriber_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported transcription backend"):
        factory.build_transcriber(
            "unknown", "small", "cpu", "int8", "ignored", "ja", True, 60
        )


def test_expand_audio_inputs_splits_a_multi_track_file(tmp_path) -> None:
    from podcastpreprocessor.cli import expand_audio_inputs
    from test_audio_io import write_two_stream_mka

    source = write_two_stream_mka(tmp_path / "two.mka")

    inputs, stream_info, stream_counts = expand_audio_inputs([str(source)])

    assert [(spec.path, spec.stream_index) for spec in inputs] == [
        (str(source), 0),
        (str(source), 1),
    ]
    assert stream_counts == {str(source): 2}
    assert stream_info[(str(source), 1)].channels == 2


def test_expand_audio_inputs_keeps_unprobable_files_as_single_input(
    tmp_path,
) -> None:
    from podcastpreprocessor.cli import expand_audio_inputs

    missing = tmp_path / "gone.wav"

    inputs, _info, stream_counts = expand_audio_inputs([str(missing)])

    assert [(spec.path, spec.stream_index) for spec in inputs] == [(str(missing), 0)]
    assert stream_counts == {str(missing): 1}


def test_main_processes_each_track_of_a_multi_track_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import opentimelineio as otio

    from podcastpreprocessor import cli
    from podcastpreprocessor.models import RawSegment, WordTimestamp
    from test_audio_io import write_two_stream_mka

    source = write_two_stream_mka(tmp_path / "two.mka")
    output = tmp_path / "out.otio"

    class FakeTranscriber:
        def transcribe_sources(
            self, audio_inputs, progress_callback=None, phase_callback=None
        ):
            results = []
            for index, spec in enumerate(audio_inputs, start=1):
                text = f"トラック{spec.stream_index}の発話"
                segment = RawSegment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text=text,
                    no_speech_prob=0.0,
                    words=[WordTimestamp(text, 0.0, 1.0, 1.0)],
                    source_id=f"source_{index}",
                )
                results.append((f"source_{index}", spec.path, [segment], 1.0))
            return results

    monkeypatch.setattr(
        cli, "build_transcriber", lambda *args, **kwargs: FakeTranscriber()
    )

    assert cli.main([str(source), str(output), "--no-cache"]) == 0

    timeline = otio.adapters.read_from_file(str(output))
    audio_tracks = [
        track for track in timeline.tracks if track.kind == otio.schema.TrackKind.Audio
    ]
    assert [track.name for track in audio_tracks] == [
        "two トラック1 音声トラック",
        "two トラック2 音声トラック",
    ]
    for expected_track_id, track in enumerate(audio_tracks):
        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert clips
        for clip in clips:
            assert clip.media_reference.target_url == str(source)
            channels = [
                dict(entry) for entry in clip.metadata["Resolve_OTIO"]["Channels"]
            ]
            assert channels == [
                {"Source Channel ID": 0, "Source Track ID": expected_track_id},
                {"Source Channel ID": 1, "Source Track ID": expected_track_id},
            ]
    subtitle_texts = [
        clip.media_reference.parameters["text"]
        for track in timeline.tracks
        for clip in track
        if isinstance(clip, otio.schema.Clip)
        and isinstance(clip.media_reference, otio.schema.GeneratorReference)
        and clip.media_reference.generator_kind == "Text"
    ]
    assert "トラック0の発話" in subtitle_texts
    assert "トラック1の発話" in subtitle_texts
