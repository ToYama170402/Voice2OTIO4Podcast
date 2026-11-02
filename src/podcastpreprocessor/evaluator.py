"""Rule-based cut-candidate evaluation."""

from dataclasses import dataclass
import re

from .models import RefinedSegment

DEFAULT_FILLERS = ("えーと", "あのー", "んー", "まー", "えっ")


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
