from podcastpreprocessor.models import ProcessedSegment
from podcastpreprocessor.srt import write_srt


def test_write_srt_formats_segments_in_timeline_order(tmp_path) -> None:
    output_path = tmp_path / "subtitles.srt"
    segments = [
        ProcessedSegment("second", 2, 45, 75, 45, 75, "次の字幕"),
        ProcessedSegment("first", 1, 0, 30, 0, 30, "最初の字幕"),
    ]

    write_srt(str(output_path), segments, fps=30)

    assert output_path.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "最初の字幕\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:02,500\n"
        "次の字幕\n"
        "\n"
    )
