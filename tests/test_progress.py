import io

import pytest

from podcastpreprocessor.progress import (
    PREPARE_FRACTION,
    PROCESSING_WEIGHT,
    TRANSCRIPTION_WEIGHT,
    ProgressReporter,
)


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_ratio_for_single_file_spans_all_phases() -> None:
    reporter = ProgressReporter(io.StringIO(), total_files=1)

    assert reporter.ratio_for(0, "prepare", 0.0) == 0.0
    assert reporter.ratio_for(0, "transcribe", 0.0) == pytest.approx(
        TRANSCRIPTION_WEIGHT * PREPARE_FRACTION
    )
    assert reporter.ratio_for(0, "transcribe", 1.0) == pytest.approx(
        TRANSCRIPTION_WEIGHT
    )
    assert reporter.ratio_for(0, "processing", 0.0) == pytest.approx(
        TRANSCRIPTION_WEIGHT
    )
    assert reporter.ratio_for(0, "processing", 1.0) == pytest.approx(1.0)


def test_ratio_for_rejects_unknown_phase() -> None:
    reporter = ProgressReporter(io.StringIO(), total_files=1)

    with pytest.raises(ValueError, match="unknown phase"):
        reporter.ratio_for(0, "unknown", 0.5)


def test_overall_progress_does_not_reset_between_files() -> None:
    reporter = ProgressReporter(io.StringIO(), total_files=2)
    ratios: list[float] = []

    reporter.phase("/audio/a.wav", "読み込み・リサンプリング")
    ratios.append(reporter.ratio)
    reporter("/audio/a.wav", 5.0, 10.0)
    ratios.append(reporter.ratio)
    reporter("/audio/a.wav", 10.0, 10.0)
    ratios.append(reporter.ratio)
    # Second file starts where the first file's slice ended (45%).
    reporter.phase("/audio/b.wav", "読み込み・リサンプリング")
    ratios.append(reporter.ratio)
    reporter("/audio/b.wav", 0.0, 10.0)
    ratios.append(reporter.ratio)
    reporter("/audio/b.wav", 10.0, 10.0)
    ratios.append(reporter.ratio)
    reporter.start_processing()
    ratios.append(reporter.ratio)
    reporter.finish()
    ratios.append(reporter.ratio)

    assert ratios == sorted(ratios)
    assert ratios[2] == pytest.approx(TRANSCRIPTION_WEIGHT / 2)
    assert ratios[3] == pytest.approx(TRANSCRIPTION_WEIGHT / 2)
    assert ratios[5] == pytest.approx(TRANSCRIPTION_WEIGHT)
    assert ratios[-1] == 1.0


def test_reporter_renders_phase_and_overall_percent_on_tty() -> None:
    stream = TTYBuffer()
    reporter = ProgressReporter(stream, total_files=2)

    reporter.phase("/audio/episode.wav", "読み込み・リサンプリング")
    reporter("/audio/episode.wav", 5.0, 10.0)

    output = stream.getvalue()
    assert "読み込み・リサンプリング" in output
    assert "文字起こし" in output
    assert "episode.wav (1/2)" in output
    # Overall ratio: half of the first file's 45% slice, past preparation.
    assert "25.9%" in output


def test_reporter_is_quiet_for_non_tty() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(stream, total_files=1)

    reporter.phase("episode.wav", "読み込み・リサンプリング")
    reporter("episode.wav", 5.0, 10.0)
    reporter.start_processing()
    reporter.finish()

    assert stream.getvalue() == ""


def test_reporter_tolerates_unknown_duration() -> None:
    stream = TTYBuffer()
    reporter = ProgressReporter(stream, total_files=1)

    reporter.phase("episode.wav", "読み込み")
    reporter("episode.wav", 0.0, None)

    assert "文字起こし" in stream.getvalue()
    assert reporter.ratio == pytest.approx(
        TRANSCRIPTION_WEIGHT * PREPARE_FRACTION * 0.0
    )


def test_reporter_requires_positive_total_files() -> None:
    with pytest.raises(ValueError, match="total_files"):
        ProgressReporter(io.StringIO(), total_files=0)


def test_processing_phase_uses_final_weight() -> None:
    reporter = ProgressReporter(io.StringIO(), total_files=3)

    reporter.start_processing()

    assert reporter.ratio == pytest.approx(TRANSCRIPTION_WEIGHT)
    assert PROCESSING_WEIGHT == pytest.approx(0.1)
