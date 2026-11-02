"""Deterministic subtitle segmentation from word timestamps."""

from collections.abc import Callable, Sequence

from sudachipy import Dictionary, SplitMode

from .models import RawSegment, RefinedSegment, WordTimestamp

BoundaryPredicate = Callable[[Sequence[WordTimestamp]], bool]

_TOKENIZER = Dictionary().tokenizer()


def _default_boundary(words: Sequence[WordTimestamp]) -> bool:
    text = _text(words)
    if not text:
        return False
    if text.endswith(("、", "。", "？", "！", ",", ".", "?", "!")):
        return True
    morphemes = _TOKENIZER.tokenize(text, SplitMode.C)
    if not morphemes:
        return False
    return morphemes[-1].part_of_speech()[0] in {
        "助詞",
        "助動詞",
        "動詞",
        "形容詞",
        "副詞",
    }


def _text(words: Sequence[WordTimestamp]) -> str:
    return "".join(word.word for word in words).strip()


def _make_segment(
    source: RawSegment,
    index: int,
    words: Sequence[WordTimestamp],
    track_index: int,
) -> RefinedSegment:
    return RefinedSegment(
        segment_id=f"{source.id}_{index}",
        track_index=track_index,
        start=words[0].start,
        end=words[-1].end,
        text=_text(words),
        no_speech_prob=source.no_speech_prob,
        words=list(words),
        source_id=source.source_id,
    )


def segment_raw(
    source: RawSegment,
    max_characters: int = 20,
    max_duration: float = 4.0,
    preferred_duration: float = 3.5,
    boundary_predicate: BoundaryPredicate = _default_boundary,
    track_index: int = 1,
) -> list[RefinedSegment]:
    """Split a raw segment while preserving every word timestamp."""
    if max_characters <= 0 or max_duration <= 0 or preferred_duration <= 0:
        raise ValueError("segmentation limits must be positive")
    if preferred_duration > max_duration:
        raise ValueError("preferred_duration cannot exceed max_duration")
    if track_index <= 0:
        raise ValueError("track_index must be positive")
    if not source.words:
        text = source.text.strip()
        if not text:
            return []
        return [
            RefinedSegment(
                f"{source.id}_0",
                1,
                source.start,
                source.end,
                text,
                source.no_speech_prob,
                source_id=source.source_id,
            )
        ]

    result: list[RefinedSegment] = []
    current: list[WordTimestamp] = []
    for word in source.words:
        candidate = [*current, word]
        candidate_text = _text(candidate)
        candidate_duration = candidate[-1].end - candidate[0].start
        exceeds_limit = (
            len(candidate_text) > max_characters or candidate_duration > max_duration
        )
        if current and exceeds_limit:
            result.append(_make_segment(source, len(result), current, track_index))
            current = [word]
            continue

        current = candidate
        reached_preferred_boundary = (
            boundary_predicate(current) and candidate_duration >= preferred_duration
        )
        if reached_preferred_boundary:
            result.append(_make_segment(source, len(result), current, track_index))
            current = []

    if current:
        result.append(_make_segment(source, len(result), current, track_index))
    return result
