import pytest

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


def test_write_srt_rejects_non_positive_fps(tmp_path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        write_srt(str(tmp_path / "x.srt"), [], fps=0)
    with pytest.raises(ValueError, match="fps must be positive"):
        write_srt(str(tmp_path / "x.srt"), [], fps=-30)


def test_write_srt_uses_30fps_by_default(tmp_path) -> None:
    output_path = tmp_path / "default.srt"
    write_srt(str(output_path), [ProcessedSegment("s", 1, 0, 30, 0, 30, "字幕")])

    assert "00:00:00,000 --> 00:00:01,000" in output_path.read_text(encoding="utf-8")


def test_write_srt_formats_hours_and_minutes(tmp_path) -> None:
    output_path = tmp_path / "long.srt"
    segments = [
        ProcessedSegment("minute", 1, 1800, 1830, 0, 30, "1分"),
        ProcessedSegment("hour", 1, 108000, 108030, 0, 30, "1時間"),
        ProcessedSegment("mixed", 1, 109830, 109860, 0, 30, "1時間1分"),
    ]

    write_srt(str(output_path), segments, fps=30)

    text = output_path.read_text(encoding="utf-8")
    assert "00:01:00,000 --> 00:01:01,000" in text
    assert "01:00:00,000 --> 01:00:01,000" in text
    assert "01:01:01,000 --> 01:01:02,000" in text


def test_write_srt_millisecond_rounding(tmp_path) -> None:
    output_path = tmp_path / "ms.srt"
    segments = [ProcessedSegment("s", 1, 1, 2, 0, 30, "x")]
    write_srt(str(output_path), segments, fps=30)
    assert "00:00:00,033 --> 00:00:00,067" in output_path.read_text(encoding="utf-8")

    output_path = tmp_path / "ms16.srt"
    segments = [ProcessedSegment("s", 1, 1, 3, 0, 30, "x")]
    write_srt(str(output_path), segments, fps=16)
    assert "00:00:00,062 --> 00:00:00,188" in output_path.read_text(encoding="utf-8")


def test_write_srt_orders_by_start_then_track_then_end(tmp_path) -> None:
    output_path = tmp_path / "order.srt"
    segments = [
        ProcessedSegment("long", 1, 0, 90, 0, 90, "long"),
        ProcessedSegment("track2", 2, 0, 30, 0, 30, "track2"),
        ProcessedSegment("short", 1, 0, 30, 0, 30, "short"),
    ]

    write_srt(str(output_path), segments, fps=30)

    text = output_path.read_text(encoding="utf-8")
    assert text.index("short") < text.index("long") < text.index("track2")


def test_write_srt_empty_segments_writes_empty_file(tmp_path) -> None:
    output_path = tmp_path / "empty.srt"
    write_srt(str(output_path), [], fps=30)
    assert output_path.read_text(encoding="utf-8") == ""
