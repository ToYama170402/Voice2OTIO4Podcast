"""FCP7 XML generation for Resolve-compatible timelines."""

from pathlib import Path
from xml.etree import ElementTree

from .models import ProcessedSegment


def _append_text(
    parent: ElementTree.Element, tag: str, text: object
) -> ElementTree.Element:
    element = ElementTree.SubElement(parent, tag)
    element.text = str(text)
    return element


def _track_indices(source_track: int) -> tuple[int, int, int]:
    if source_track <= 0:
        raise ValueError("source_track must be positive")
    return source_track * 2 - 1, source_track * 2, source_track


def _links(
    clip: ElementTree.Element,
    segment: ProcessedSegment,
    subtitle_id: str,
    audio_id: str,
) -> None:
    subtitle_track, flag_track, audio_track = _track_indices(segment.track_index)
    links = [(subtitle_id, "video", subtitle_track)]
    if segment.is_cut_candidate:
        links.insert(0, (f"flag_{segment.segment_id}", "video", flag_track))
    links.append((audio_id, "audio", audio_track))
    for link_id, media_type, track_index in links:
        link = ElementTree.SubElement(clip, "link")
        _append_text(link, "linkclipid", link_id)
        _append_text(link, "mediatype", media_type)
        _append_text(link, "trackindex", track_index)
        _append_text(link, "clipitemid", link_id)


def _base_clip(
    parent: ElementTree.Element,
    clip_id: str,
    name: str,
    start: int,
    end: int,
    source_in: int,
    source_out: int,
) -> ElementTree.Element:
    clip = ElementTree.SubElement(parent, "clipitem", {"id": clip_id})
    _append_text(clip, "name", name)
    _append_text(clip, "start", start)
    _append_text(clip, "end", end)
    _append_text(clip, "in", source_in)
    _append_text(clip, "out", source_out)
    return clip


def _append_flag_clip(track: ElementTree.Element, segment: ProcessedSegment) -> None:
    clip_id = f"flag_{segment.segment_id}"
    subtitle_id = f"sub_{segment.segment_id}"
    audio_id = f"audio_{segment.segment_id}"
    clip = _base_clip(
        track,
        clip_id,
        segment.flag_label or "[要カット]",
        segment.start_frame,
        segment.end_frame,
        0,
        segment.end_frame - segment.start_frame,
    )
    labels = ElementTree.SubElement(clip, "labels")
    _append_text(labels, "label2", segment.color_label or "Red")
    marker = ElementTree.SubElement(clip, "marker")
    _append_text(marker, "name", "カット候補")
    _append_text(marker, "comment", segment.marker_reason)
    _append_text(marker, "in", 0)
    _append_text(marker, "out", 0)
    _links(clip, segment, subtitle_id, audio_id)


def _append_subtitle_clip(
    track: ElementTree.Element, segment: ProcessedSegment
) -> None:
    clip_id = f"sub_{segment.segment_id}"
    _base = _base_clip(
        track,
        clip_id,
        segment.subtitle_text,
        segment.start_frame,
        segment.end_frame,
        0,
        segment.end_frame - segment.start_frame,
    )
    _links(_base, segment, clip_id, f"audio_{segment.segment_id}")


def _append_audio_clip(
    track: ElementTree.Element,
    segment: ProcessedSegment,
    audio_path: str,
) -> None:
    clip_id = f"audio_{segment.segment_id}"
    clip = _base_clip(
        track,
        clip_id,
        f"Audio_{segment.segment_id}",
        segment.start_frame,
        segment.end_frame,
        segment.in_frame,
        segment.out_frame,
    )
    file_element = ElementTree.SubElement(clip, "file", {"id": "file_audio"})
    _append_text(file_element, "pathurl", Path(audio_path).resolve().as_uri())
    _links(clip, segment, f"sub_{segment.segment_id}", clip_id)


def generate_xmeml(
    segments: list[ProcessedSegment],
    audio_path: str,
    sequence_name: str = "Podcast_Timeline",
    fps: int = 30,
) -> ElementTree.ElementTree:
    """Build an xmeml document from processed, frame-based segments."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not segments:
        raise ValueError("at least one segment is required")

    root = ElementTree.Element("xmeml", {"version": "5"})
    sequence = ElementTree.SubElement(root, "sequence")
    _append_text(sequence, "name", sequence_name)
    rate = ElementTree.SubElement(sequence, "rate")
    _append_text(rate, "timebase", fps)
    _append_text(rate, "ntsc", "FALSE")
    media = ElementTree.SubElement(sequence, "media")
    video = ElementTree.SubElement(media, "video")
    audio = ElementTree.SubElement(media, "audio")

    max_track = max(segment.track_index for segment in segments)
    subtitle_tracks: dict[int, ElementTree.Element] = {}
    flag_tracks: dict[int, ElementTree.Element] = {}
    for track_index in range(1, max_track + 1):
        subtitle_tracks[track_index] = ElementTree.SubElement(video, "track")
        flag_tracks[track_index] = ElementTree.SubElement(video, "track")
    audio_tracks = {
        track_index: ElementTree.SubElement(audio, "track")
        for track_index in range(1, max_track + 1)
    }

    for segment in segments:
        _append_subtitle_clip(subtitle_tracks[segment.track_index], segment)
        if segment.is_cut_candidate:
            _append_flag_clip(flag_tracks[segment.track_index], segment)
        _append_audio_clip(audio_tracks[segment.track_index], segment, audio_path)

    return ElementTree.ElementTree(root)


def write_xmeml(
    output_path: str,
    segments: list[ProcessedSegment],
    audio_path: str,
    sequence_name: str = "Podcast_Timeline",
    fps: int = 30,
) -> None:
    """Write a generated xmeml document with an XML declaration."""
    document = generate_xmeml(segments, audio_path, sequence_name, fps)
    ElementTree.indent(document, space="  ")
    document.write(output_path, encoding="utf-8", xml_declaration=True)
