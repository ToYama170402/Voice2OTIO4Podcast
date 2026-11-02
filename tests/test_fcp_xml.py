from xml.etree import ElementTree

from podcastpreprocessor.fcp_xml import generate_xmeml
from podcastpreprocessor.models import ProcessedSegment


def test_generate_xmeml_contains_tracks_links_and_candidate_marker() -> None:
    segments = [
        ProcessedSegment("t1_c1", 1, 0, 30, 0, 30, "こんにちは"),
        ProcessedSegment(
            "t1_c2",
            1,
            30,
            60,
            30,
            60,
            "えーと、本題です",
            is_cut_candidate=True,
            flag_label="[要カット: フィラー]",
            marker_reason="判定: フィラー (えーと)",
            color_label="Red",
        ),
    ]

    document = generate_xmeml(segments, "speaker_a.wav")
    root = document.getroot()

    assert root.tag == "xmeml"
    assert len(root.findall("./sequence/media/video/track")) == 2
    assert len(root.findall("./sequence/media/audio/track")) == 1
    assert root.find(".//clipitem[@id='flag_t1_c1']") is None
    flag = root.find(".//clipitem[@id='flag_t1_c2']")
    assert flag is not None
    assert flag.findtext("name") == "[要カット: フィラー]"
    assert flag.findtext("labels/label2") == "Red"
    assert flag.findtext("marker/comment") == "判定: フィラー (えーと)"

    clip_ids = {clip.get("id") for clip in root.findall(".//clipitem")}
    for clip in root.findall(".//clipitem"):
        link_ids = {link.findtext("linkclipid") for link in clip.findall("link")}
        assert link_ids <= clip_ids


def test_generate_xmeml_is_serializable() -> None:
    document = generate_xmeml(
        [ProcessedSegment("t1_c1", 1, 0, 30, 0, 30, "text")],
        "C:/audio/speaker.wav",
    )
    ElementTree.fromstring(ElementTree.tostring(document.getroot()))