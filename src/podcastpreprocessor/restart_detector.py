"""Restart (言い直し) detection over the transcription token stream.

The token (word timestamp) stream is scanned directly, before
segmentation. Compared with matching on segment text, this pins the
abandoned attempt to exact token times, so an attempt that sits inside
a longer segment can be split out on its own instead of flagging the
whole segment.

Two patterns are detected:

- An utterance whose opening is immediately re-spoken by the next
  utterance, in full or in large part (the first utterance is the
  abandoned attempt).
- An utterance that immediately repeats its own opening phrase and
  then continues (only the first copy is the attempt).

Backchannels, set agreement phrases, emphatic repetition and complete
sentences are excluded, mirroring the precision-first rules the
segment-text version used.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .evaluator import is_filler_text
from .models import RawSegment

# Backchannels and set agreement phrases repeat naturally across
# adjacent utterances; they are never treated as abandoned attempts.
_EXCLUSIONS = frozenset(
    {
        "はい",
        "うん",
        "そう",
        "そうだね",
        "そうですね",
        "なるほど",
        "たしかに",
        "確かに",
        "うんうん",
        "はいはい",
        "そうそう",
        "あれ",
    }
)

_SENTENCE_FINAL = frozenset("。！？!?")
_STRIP = re.compile(r"[\s、。，．！？!?,.・…「」『』（）()]")

_MIN_OUTLIER_CHARS = 3
_LONG_TOKEN_S = 0.8
_MIN_ATTEMPT_CHARS = 3
_MAX_OVERLAP_S = 0.5
_MIN_FULL_REPEAT_CHARS = 6


@dataclass(frozen=True)
class RestartRegion:
    """The time range of an abandoned attempt, in seconds."""

    start: float
    end: float
    text: str

    @property
    def reason(self) -> str:
        return f"言い直し ({self.text})"


@dataclass
class _Char:
    char: str
    start: float
    end: float
    word_start: float
    word_end: float


def _effective_end(chars: list[_Char]) -> float:
    """End of an utterance, discounting a pause absorbed by its last token.

    Some backends tile token times, so a pause after the utterance is
    folded into the last token's duration. When that duration is an
    outlier against the utterance's other tokens, the speech itself
    is taken to end one typical token after the last token's start.
    """
    last = chars[-1]
    if len(chars) < _MIN_OUTLIER_CHARS:
        return last.end
    durations = sorted(char.end - char.start for char in chars[:-1])
    median = durations[len(durations) // 2]
    if last.end - last.start > _LONG_TOKEN_S and last.end - last.start > 2.5 * max(
        median, 0.01
    ):
        return last.start + median
    return last.end


@dataclass
class _Utterance:
    chars: list[_Char] = field(default_factory=list)
    complete: bool = False

    @property
    def text(self) -> str:
        return "".join(char.char for char in self.chars)

    @property
    def start(self) -> float:
        return self.chars[0].start

    @property
    def end(self) -> float:
        return _effective_end(self.chars)


def _build_utterances(
    segments: Sequence[RawSegment],
    pause_gap: float,
    pause_token_duration: float,
) -> list[_Utterance]:
    """Chunk the token stream into utterances.

    A new utterance starts at a raw segment boundary, after a
    sentence-final punctuation token, after a real time gap, and after
    a long-held token (backends such as SenseVoice absorb the pause
    into the previous token's duration instead of leaving a gap).
    Punctuation and whitespace never become utterance characters, but
    sentence-final punctuation marks the utterance as complete.
    """
    utterances: list[_Utterance] = []
    current: _Utterance | None = None
    previous: _Char | None = None
    previous_segment = -1
    for segment_index, segment in enumerate(segments):
        for word in segment.words:
            for char in word.word:
                if char in _SENTENCE_FINAL:
                    if current is not None:
                        current.complete = True
                    continue
                if _STRIP.match(char):
                    continue
                token = _Char(char, word.start, word.end, word.start, word.end)
                boundary = (
                    current is None
                    or segment_index != previous_segment
                    or (
                        previous is not None
                        and previous.end - previous.start >= pause_token_duration
                    )
                    or (previous is not None and token.start - previous.end > pause_gap)
                    or (current is not None and current.complete and current.chars)
                )
                if boundary or current is None:
                    current = _Utterance()
                    utterances.append(current)
                current.chars.append(token)
                previous = token
                previous_segment = segment_index
    return [utterance for utterance in utterances if utterance.chars]


def _common_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b, strict=False):
        if char_a != char_b:
            break
        length += 1
    return length


def _is_repeated_exclusion(core: str) -> bool:
    """Return True when the text is one backchannel repeated over and over."""
    return any(
        len(core) > len(unit)
        and len(core) % len(unit) == 0
        and core == unit * (len(core) // len(unit))
        for unit in _EXCLUSIONS
    )


def _repeat_region(utterance: _Utterance) -> RestartRegion | None:
    """Region of the first copy when an utterance repeats its opening.

    The repetition must be followed by more speech: a repetition that
    makes up the whole utterance (してないしてない) is emphasis.
    """
    core = utterance.text
    best: int | None = None
    for width in range(4, len(core) // 2 + 1):
        unit = core[:width]
        if len(set(unit)) == 1 or unit in _EXCLUSIONS:
            continue
        if core[width : 2 * width] == unit and len(core) > 2 * width:
            best = width
    if best is None:
        return None
    first = utterance.chars[0]
    last = utterance.chars[best - 1]
    return RestartRegion(
        start=first.word_start,
        end=last.word_end,
        text=f"{core[:best]} の繰り返し",
    )


def _pair_region(
    attempt: _Utterance,
    following: _Utterance,
    max_gap: float,
    max_attempt_duration: float,
    min_prefix: int,
    min_coverage: float,
) -> RestartRegion | None:
    """Region of an utterance whose opening the next utterance re-speaks."""
    core = attempt.text
    if (
        len(core) < _MIN_ATTEMPT_CHARS
        or core in _EXCLUSIONS
        or _is_repeated_exclusion(core)
        or is_filler_text(core)
    ):
        return None
    gap = following.start - attempt.end
    if gap < -_MAX_OVERLAP_S or gap > max_gap:
        return None
    if attempt.end - attempt.start > max_attempt_duration:
        return None
    prefix = _common_prefix_length(core, following.text)
    if prefix == 0:
        return None
    full = prefix == len(core)
    full_repeat = (
        full
        and len(core) >= min_prefix
        and (not attempt.complete or len(core) >= _MIN_FULL_REPEAT_CHARS)
    )
    partial_repeat = (
        not attempt.complete
        and prefix >= min_prefix
        and prefix >= min_coverage * len(core)
    )
    if full_repeat or (full and not attempt.complete) or partial_repeat:
        flagged = True
    else:
        flagged = False
    if not flagged:
        return None
    return RestartRegion(
        start=attempt.start,
        end=attempt.end,
        text=core[:prefix][:14],
    )


def token_restart_regions(
    segments: Sequence[RawSegment],
    max_gap: float = 4.0,
    max_attempt_duration: float = 6.0,
    min_prefix: int = 4,
    min_coverage: float = 0.6,
    pause_gap: float = 0.4,
    pause_token_duration: float = 0.8,
) -> list[RestartRegion]:
    """Detect abandoned attempts in one source's token stream.

    Returns the attempt regions in time order. Segments without word
    timestamps yield no regions.
    """
    segments = [segment for segment in segments if segment.words]
    if not segments:
        return []
    utterances = _build_utterances(segments, pause_gap, pause_token_duration)
    regions: list[RestartRegion] = []
    for index, utterance in enumerate(utterances):
        repeat = _repeat_region(utterance)
        if repeat is not None:
            regions.append(repeat)
        if index + 1 < len(utterances):
            pair = _pair_region(
                utterance,
                utterances[index + 1],
                max_gap,
                max_attempt_duration,
                min_prefix,
                min_coverage,
            )
            if pair is not None:
                regions.append(pair)
    regions.sort(key=lambda region: region.start)
    kept: list[RestartRegion] = []
    for region in regions:
        overlap = any(
            min(region.end, other.end) - max(region.start, other.start)
            > 0.5 * min(region.end - region.start, other.end - other.start)
            for other in kept
        )
        if not overlap:
            kept.append(region)
    return kept
