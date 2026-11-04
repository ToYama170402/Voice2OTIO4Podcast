"""OpenTimelineIO generation for DaVinci Resolve."""

from collections.abc import Iterable
from typing import Any

import opentimelineio as otio

from .models import AudioSource, ProcessedSegment
from .timeline import build_audio_slices


def _metadata(link_group_id: int, **extra: object) -> dict[str, object]:
    resolve = {"Link Group ID": link_group_id, **extra}
    return {"Resolve_OTIO": resolve}


def _clip(
    name: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    link_group_id: int,
    media_reference: object | None = None,
    metadata: dict[str, object] | None = None,
) -> otio.schema.Clip:
    clip = otio.schema.Clip(
        name=name,
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frame, fps),
            duration=otio.opentime.RationalTime(end_frame - start_frame, fps),
        ),
        media_reference=media_reference,
    )
    clip.metadata.update(metadata or _metadata(link_group_id))
    return clip


def _title_clip(
    text: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    link_group_id: int,
    available_duration_frames: int,
    metadata: dict[str, object] | None = None,
) -> otio.schema.Clip:
    """Create a generated text clip instead of a clip with missing media."""
    reference = otio.schema.GeneratorReference(
        generator_kind="Text",
        parameters={"text": text},
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime(available_duration_frames, fps),
        ),
    )
    return _clip(
        text,
        0,
        end_frame - start_frame,
        fps,
        link_group_id,
        reference,
        metadata,
    )


def _source_map(
    segments: Iterable[ProcessedSegment],
    audio_sources: Iterable[AudioSource],
    fps: float,
) -> dict[str, AudioSource]:
    sources = {source.source_id: source for source in audio_sources}
    for segment in segments:
        if segment.source_id not in sources:
            if not segment.audio_path:
                raise ValueError(f"missing audio source for {segment.source_id}")
            sources[segment.source_id] = AudioSource(
                segment.source_id,
                segment.audio_path,
                segment.track_index,
                segment.end_frame,
            )
    if not sources:
        raise ValueError("at least one audio source is required")
    return sources


def _append_at_time(
    track: otio.schema.Track,
    item: Any,
    start_frame: int,
    cursor: int,
    fps: float,
) -> int:
    if start_frame < cursor:
        raise ValueError("track items overlap after frame conversion")
    if start_frame > cursor:
        track.append(
            otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    duration=otio.opentime.RationalTime(start_frame - cursor, fps)
                )
            )
        )
    track.append(item)
    return start_frame + int(item.duration().value)


def generate_otio(
    segments: Iterable[ProcessedSegment],
    audio_sources: Iterable[AudioSource] = (),
    sequence_name: str = "Podcast_Timeline",
    fps: int = 30,
) -> otio.schema.Timeline:
    """Build a Resolve-compatible timeline with linked clips."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    segment_list = list(segments)
    if not segment_list:
        raise ValueError("at least one segment is required")
    sources = _source_map(segment_list, audio_sources, fps)
    timeline = otio.schema.Timeline(name=sequence_name)
    timeline.metadata["Resolve_OTIO"] = {"Resolve OTIO Meta Version": "1.0"}

    next_link_id = 1
    for source in sorted(sources.values(), key=lambda item: item.track_index):
        marker_track = otio.schema.Track(
            name=f"{source.name or source.source_id} 目印トラック",
            kind=otio.schema.TrackKind.Video,
        )
        subtitle_track = otio.schema.Track(
            name=f"{source.name or source.source_id} 字幕トラック",
            kind=otio.schema.TrackKind.Video,
        )
        audio_track = otio.schema.Track(
            name=f"{source.name or source.source_id} 音声トラック",
            kind=otio.schema.TrackKind.Audio,
        )
        source_segments = [
            segment for segment in segment_list if segment.source_id == source.source_id
        ]
        marker_cursor = 0
        subtitle_cursor = 0
        for segment in sorted(source_segments, key=lambda item: item.start_frame):
            link_group_id = segment.link_group_id or next_link_id
            next_link_id = max(next_link_id, link_group_id + 1)
            segment.link_group_id = link_group_id
            common = _metadata(link_group_id)
            subtitle_cursor = _append_at_time(
                subtitle_track,
                _title_clip(
                    segment.subtitle_text,
                    segment.start_frame,
                    segment.end_frame,
                    fps,
                    link_group_id,
                    source.duration_frames,
                    metadata=common,
                ),
                segment.start_frame,
                subtitle_cursor,
                fps,
            )
            if segment.is_cut_candidate:
                marker_clip = _title_clip(
                    segment.flag_label or "[要カット]",
                    segment.start_frame,
                    segment.end_frame,
                    fps,
                    link_group_id,
                    source.duration_frames,
                    metadata={
                        **common,
                        "cut_candidate": True,
                        "marker_reason": segment.marker_reason,
                        "color_label": segment.color_label or "Red",
                    },
                )
                marker_clip.markers.append(
                    otio.schema.Marker(
                        name="カット候補",
                        marked_range=marker_clip.source_range,
                    )
                )
                marker_cursor = _append_at_time(
                    marker_track,
                    marker_clip,
                    segment.start_frame,
                    marker_cursor,
                    fps,
                )

        reference = otio.schema.ExternalReference(target_url=source.path)
        for audio_slice in build_audio_slices(
            segment_list, source.source_id, source.duration_frames
        ):
            link_group_id = next(
                (
                    segment.link_group_id
                    for segment in source_segments
                    if segment.segment_id == audio_slice.segment_id
                ),
                0,
            )
            audio_track.append(
                _clip(
                    source.name or source.source_id,
                    audio_slice.source_in_frame,
                    audio_slice.source_out_frame,
                    fps,
                    link_group_id,
                    reference,
                    (
                        _metadata(link_group_id)
                        if link_group_id
                        else {"source_slice": True}
                    ),
                )
            )
        timeline.tracks.append(marker_track)
        timeline.tracks.append(subtitle_track)
        timeline.tracks.append(audio_track)
    return timeline


def write_otio(
    output_path: str,
    segments: Iterable[ProcessedSegment],
    audio_sources: Iterable[AudioSource] = (),
    sequence_name: str = "Podcast_Timeline",
    fps: int = 30,
) -> None:
    timeline = generate_otio(segments, audio_sources, sequence_name, fps)
    otio.adapters.write_to_file(timeline, output_path)
