"""Rule-based cut-candidate evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
import re

from .models import RefinedSegment

DEFAULT_FILLERS = ("えーと", "あのー", "んー", "まー", "えっ")

_FILLER_PUNCTUATION = "、。，．！？!?,.・…"

# Spoken filler forms as a whole utterance, including variants the plain
# substring list misses (えっと, あの, えー, そのー, うーん, ...). Applied to
# a segment's filler core, so ordinary speech is never matched.
FILLER_PATTERN = re.compile(
    r"^(?:えー+と|えー+|えっと|あのー*|ああ|あ|んー+|ん|まー+|まあ|うーん|うん|あー+|えっ|そのー*)$"
)


def filler_core(text: str) -> str:
    """Reduce text to its filler core by dropping whitespace and punctuation."""
    return re.sub(r"\s+", "", text).strip(_FILLER_PUNCTUATION)


def is_filler_text(text: str) -> bool:
    """Return True when the whole text is a single filler utterance."""
    return bool(FILLER_PATTERN.match(filler_core(text)))


@dataclass(frozen=True)
class Evaluation:
    is_cut_candidate: bool
    reasons: tuple[str, ...]

    @property
    def flag_label(self) -> str:
        return f"[要カット: {' / '.join(self.reasons)}]" if self.reasons else ""

    @property
    def marker_reason(self) -> str:
        return f"判定: {' / '.join(self.reasons)}" if self.reasons else ""


def evaluate_segment(
    segment: RefinedSegment,
    previous_end: float | None = None,
    pause_threshold: float = 0.4,
    no_speech_threshold: float = 0.8,
    fillers: tuple[str, ...] = DEFAULT_FILLERS,
) -> Evaluation:
    """Evaluate one segment against the configured cut rules."""
    if pause_threshold < 0:
        raise ValueError("pause_threshold must be non-negative")
    if not 0 <= no_speech_threshold <= 1:
        raise ValueError("no_speech_threshold must be between 0 and 1")

    reasons: list[str] = []
    if previous_end is not None and segment.start - previous_end > pause_threshold:
        reasons.append(f"無音 {segment.start - previous_end:.1f}秒")

    matched_fillers = [filler for filler in fillers if filler in segment.text]
    if matched_fillers:
        reasons.append(f"フィラー ({'、'.join(matched_fillers)})")
    elif is_filler_text(segment.text):
        reasons.append(f"フィラー ({filler_core(segment.text)})")

    if segment.no_speech_prob > no_speech_threshold:
        reasons.append(f"非音声確率 {segment.no_speech_prob:.2f}")

    return Evaluation(bool(reasons), tuple(reasons))


def evaluate_text_pattern(
    segment: RefinedSegment,
    pattern: str,
    previous_end: float | None = None,
    **kwargs: object,
) -> Evaluation:
    """Apply the standard rules plus a caller-provided filler pattern."""
    if not re.search(pattern, segment.text):
        return evaluate_segment(segment, previous_end=previous_end, **kwargs)
    evaluation = evaluate_segment(segment, previous_end=previous_end, **kwargs)
    reasons = (*evaluation.reasons, f"フィラー ({segment.text})")
    return Evaluation(True, reasons)


# Backchannels and set agreement phrases repeat naturally across adjacent
# segments; they are never treated as abandoned attempts.
_RESTART_EXCLUSIONS = frozenset(
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

_SENTENCE_FINAL = "。！？!?"

# Punctuation and whitespace are dropped everywhere (not just at the
# edges) so that recognition differences in punctuation do not break
# the repetition matching below.
_RESTART_STRIP = re.compile(r"[\s、。，．！？!?,.・…「」『』（）()]")


def _restart_core(text: str) -> str:
    """Reduce text to the bare character sequence used for restart matching."""
    return _RESTART_STRIP.sub("", text)


def _common_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


def _is_complete_utterance(text: str) -> bool:
    """Return True when the text ends like a finished sentence."""
    return text.rstrip().rstrip("」』）)").endswith(tuple(_SENTENCE_FINAL))


def _within_segment_repeat(core: str) -> str | None:
    """Return the repeated opening unit when a segment restarts itself.

    Looks for an opening phrase of at least four characters that is
    immediately repeated and then followed by more speech, as in
    どのぐらいだっけどのぐらいだっけ話したの. A repetition that makes
    up the whole segment (してないしてない) is emphasis, not a restart,
    and backchannel units are excluded for the same reason.
    """
    best: str | None = None
    for width in range(4, len(core) // 2 + 1):
        unit = core[:width]
        if len(set(unit)) == 1 or unit in _RESTART_EXCLUSIONS:
            continue
        if core[width : 2 * width] == unit and len(core) > 2 * width:
            best = unit
    return best


def _is_repeated_exclusion(core: str) -> bool:
    """Return True when the text is one backchannel repeated over and over."""
    return any(
        len(core) > len(unit)
        and len(core) % len(unit) == 0
        and core == unit * (len(core) // len(unit))
        for unit in _RESTART_EXCLUSIONS
    )


def detect_restarts(
    segments: Sequence[RefinedSegment],
    max_gap: float = 4.0,
    max_attempt_duration: float = 6.0,
    min_prefix: int = 4,
    min_coverage: float = 0.6,
) -> dict[str, str]:
    """Flag segments that look like abandoned attempts (言い直し).

    A segment is flagged when the following segment re-speaks it, either
    in full or by repeating most of its opening before continuing, or
    when the segment itself immediately repeats its opening phrase.
    Returns a mapping of segment_id to the cut-candidate reason.
    Complete sentences and backchannels are excluded so that ordinary
    topic continuations are not flagged.
    """
    reasons: dict[str, str] = {}
    cores = [_restart_core(segment.text) for segment in segments]
    for index, segment in enumerate(segments):
        core = cores[index]
        if (
            len(core) < 3
            or core in _RESTART_EXCLUSIONS
            or _is_repeated_exclusion(core)
            or is_filler_text(segment.text)
        ):
            continue
        unit = _within_segment_repeat(core)
        if unit is not None:
            reasons[segment.segment_id] = f"言い直し ({unit} の繰り返し)"
            continue
        # The following substantive segment: empty fragments left by
        # recognition (a lone 。 with no characters) are not speech and
        # do not break the attempt / re-speak pairing.
        following: RefinedSegment | None = None
        next_core = ""
        for ahead in range(index + 1, min(index + 4, len(segments))):
            if cores[ahead]:
                following = segments[ahead]
                next_core = cores[ahead]
                break
        if following is None:
            continue
        gap = following.start - segment.end
        if gap < -0.5 or gap > max_gap:
            continue
        if segment.end - segment.start > max_attempt_duration:
            continue
        prefix = _common_prefix_length(core, next_core)
        if prefix == 0:
            continue
        complete = _is_complete_utterance(segment.text)
        full = prefix == len(core)
        if full and len(core) >= min_prefix and (not complete or len(core) >= 6):
            flagged = True
        elif full and not complete:
            flagged = True
        elif (
            not complete
            and prefix >= min_prefix
            and prefix >= min_coverage * len(core)
        ):
            flagged = True
        else:
            flagged = False
        if flagged:
            shared = core[:prefix]
            reasons[segment.segment_id] = f"言い直し ({shared[:14]})"
    return reasons
