import io

from podcastpreprocessor.cli import _ProgressDisplay


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_progress_display_renders_tty_bar() -> None:
    stream = TTYBuffer()

    _ProgressDisplay(stream)("/audio/episode.wav", 5.0, 10.0)

    assert "Transcribing episode.wav:" in stream.getvalue()
    assert "50.0%" in stream.getvalue()


def test_progress_display_is_quiet_for_non_tty() -> None:
    stream = io.StringIO()

    _ProgressDisplay(stream)("episode.wav", 5.0, 10.0)

    assert stream.getvalue() == ""
