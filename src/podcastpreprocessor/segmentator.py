"""Deterministic subtitle segmentation from word timestamps.

Segment boundaries are only ever placed at bunsetsu (Japanese clause unit)
boundaries, so a slice never cuts through the middle of a word or bunsetsu.
Filler utterances are always emitted as standalone segments, which makes
them independent slices in the OTIO timeline that an editor can cut without
touching the surrounding speech.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sudachipy import Dictionary, SplitMode

from .evaluator import FILLER_PATTERN, filler_core
from .models import RawSegment, RefinedSegment, WordTimestamp

BoundaryPredicate = Callable[[Sequence[WordTimestamp]], bool]

_TOKENIZER = Dictionary().tokenizer()

# Morpheme categories that can head a bunsetsu. Function words (particles,
# auxiliaries, suffixes, punctuation) always attach to the preceding unit.
_POS_CONJUGATION_INDEX = 4

_BUNSETSU_HEAD_POS = {
    "名詞",
    "代名詞",
    "動詞",
    "形容詞",
    "形状詞",
    "副詞",
    "連体詞",
    "接続詞",
    "感動詞",
    "接頭辞",
}

# A noun or pronoun directly following a noun, pronoun, adnominal or prefix
# continues the same bunsetsu (compound nouns such as 字幕テキスト, and
# adnominal modification such as その企画).
_NOUN_CHAIN_PREV_POS = {"名詞", "代名詞", "連体詞", "接頭辞"}

_FILLER_PATTERN = FILLER_PATTERN


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


def _is_filler_morpheme(morpheme: Any) -> bool:
    pos = morpheme.part_of_speech()
    if pos[0] != "感動詞":
        return False
    return pos[1] == "フィラー" or bool(
        _FILLER_PATTERN.match(filler_core(morpheme.surface()))
    )


def _filler_char_spans(morphemes: Sequence[Any]) -> list[tuple[int, int]]:
    """Character spans of filler phrases, e.g. えー + と + 、 in えーと、."""
    morphemes = list(morphemes)
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(morphemes):
        if not _is_filler_morpheme(morphemes[index]):
            index += 1
            continue
        end = index
        while end + 1 < len(morphemes):
            following = morphemes[end + 1]
            if following.part_of_speech()[0] not in {"助詞", "助動詞", "補助記号"}:
                break
            combined = filler_core(
                "".join(m.surface() for m in morphemes[index : end + 2])
            )
            if not _FILLER_PATTERN.match(combined):
                break
            end += 1
        spans.append((morphemes[index].begin(), morphemes[end].end()))
        index = end + 1
    return spans


def _is_bunsetsu_head(morpheme: Any, previous: Any) -> bool:
    pos = morpheme.part_of_speech()
    if pos[0] not in _BUNSETSU_HEAD_POS:
        return False
    prev_pos = previous.part_of_speech()
    if pos[0] in {"名詞", "代名詞"} and prev_pos[0] in _NOUN_CHAIN_PREV_POS:
        return False
    if pos[1] == "非自立可能" and pos[0] in {"動詞", "形容詞"}:
        # Non-independent verbs/adjectives lean on the preceding unit only
        # when they follow a content word they attach to (ている, てくる),
        # a て/で conjunctive particle, or a sahen noun (収録+し). Otherwise
        # they head their own unit (あったら, 時間かかる, あの上げる).
        if prev_pos[0] in {"動詞", "形容詞"}:
            return False
        if previous.surface() in {"て", "で"}:
            return False
        conjugation = (
            pos[_POS_CONJUGATION_INDEX] if len(pos) > _POS_CONJUGATION_INDEX else ""
        )
        if prev_pos[0] == "名詞" and "サ行変格" in conjugation:
            return False
    return True


def _bunsetsu_head_offsets(morphemes: Sequence[Any], text_length: int) -> set[int]:
    """Character offsets at which a new bunsetsu may start."""
    heads = {0, text_length}
    previous: Any = None
    for morpheme in morphemes:
        if previous is not None and _is_bunsetsu_head(morpheme, previous):
            heads.add(morpheme.begin())
        previous = morpheme
    return heads


@dataclass
class _Unit:
    """Words that must stay in one segment: one bunsetsu or one filler."""

    words: list[WordTimestamp]
    is_filler: bool


# Sudachi rejects inputs longer than 49149 bytes, so a whole-source
# transcript (an hour of speech is ~50k characters) cannot be tokenized in
# one call. Tokenize in chunks of words instead: soft chunks close at a
# sentence end or a pause once they pass the soft limit, and never exceed
# the hard limit (10000 chars stay below the byte limit even at 4 bytes
# per character).
_TOKENIZE_SOFT_LIMIT = 6000
_TOKENIZE_HARD_LIMIT = 10000
_CHUNK_PAUSE_SECONDS = 0.8


def _tokenize_chunks(
    words: Sequence[WordTimestamp],
) -> tuple[set[int], list[tuple[int, int]]]:
    """Bunsetsu head offsets and filler spans over the whole word stream."""
    heads: set[int] = set()
    filler_spans: list[tuple[int, int]] = []
    chunk_start = 0
    chunk_chars = 0
    char_offset = 0

    def flush(end_index: int) -> None:
        nonlocal chunk_start, chunk_chars
        chunk_words = words[chunk_start:end_index]
        if not chunk_words:
            return
        text = "".join(word.word for word in chunk_words)
        start_offset = char_offset - len(text)
        morphemes = _TOKENIZER.tokenize(text, SplitMode.A)
        heads.update(
            start_offset + head for head in _bunsetsu_head_offsets(morphemes, len(text))
        )
        filler_spans.extend(
            (start_offset + begin, start_offset + end)
            for begin, end in _filler_char_spans(morphemes)
        )
        chunk_start = end_index
        chunk_chars = 0

    for index, word in enumerate(words):
        chunk_chars += len(word.word)
        char_offset += len(word.word)
        if index + 1 < len(words):
            sentence_end = word.word.rstrip().endswith(("。", "！", "？", "!", "?"))
            pause = words[index + 1].start - word.end >= _CHUNK_PAUSE_SECONDS
            if chunk_chars >= _TOKENIZE_HARD_LIMIT or (
                chunk_chars >= _TOKENIZE_SOFT_LIMIT and (sentence_end or pause)
            ):
                flush(index + 1)
    flush(len(words))
    return heads, filler_spans


def _build_units(words: Sequence[WordTimestamp]) -> list[_Unit]:
    """Group words into indivisible bunsetsu units and filler units.

    A word whose timestamps straddle a filler phrase or a bunsetsu head
    cannot be split, so it stays whole inside a single normal unit.
    """
    text = "".join(word.word for word in words)
    heads, filler_spans = _tokenize_chunks(words)
    heads.add(len(text))

    # Whisper words can carry leading/trailing whitespace (" え"). Whitespace
    # is not speech, so align word spans with the trimmed text; otherwise a
    # word that starts a filler or bunsetsu no longer lines up with the
    # morpheme spans and the unit is split at the wrong place.
    word_spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words:
        leading = len(word.word) - len(word.word.lstrip())
        trailing = len(word.word) - len(word.word.rstrip())
        word_spans.append((cursor + leading, cursor + len(word.word) - trailing))
        cursor += len(word.word)

    def is_filler_word(index: int) -> bool:
        begin, end = word_spans[index]
        return any(begin >= start and end <= stop for start, stop in filler_spans)

    units: list[_Unit] = []
    current: list[WordTimestamp] = []
    current_is_filler = False

    def close_current() -> None:
        nonlocal current
        if current:
            units.append(_Unit(list(current), current_is_filler))
            current = []

    for index, word in enumerate(words):
        filler = is_filler_word(index)
        if current and filler != current_is_filler:
            close_current()
        if not current:
            current_is_filler = filler
        current.append(word)
        if filler:
            continue
        word_end = word_spans[index][1]
        if word_end in heads:
            close_current()
    close_current()
    return units


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
    """Split a raw segment at bunsetsu boundaries, isolating fillers.

    Length and duration limits are applied between units, never inside one:
    a unit that exceeds a limit on its own is emitted whole rather than cut.
    """
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

    def flush() -> None:
        nonlocal current
        if current:
            result.append(_make_segment(source, len(result), current, track_index))
            current = []

    def duration(words: Sequence[WordTimestamp]) -> float:
        return words[-1].end - words[0].start

    for unit in _build_units(source.words):
        if unit.is_filler:
            flush()
            result.append(_make_segment(source, len(result), unit.words, track_index))
            continue
        if current and (
            len(_text([*current, *unit.words])) > max_characters
            or duration([*current, *unit.words]) > max_duration
        ):
            flush()
        current = [*current, *unit.words]
        if len(_text(current)) > max_characters or duration(current) > max_duration:
            # A single unit that exceeds the limits on its own: keep it whole.
            flush()
            continue
        if boundary_predicate(current) and duration(current) >= preferred_duration:
            flush()

    flush()
    return result


def segment_source(
    raw_segments: Sequence[RawSegment],
    max_characters: int = 20,
    max_duration: float = 4.0,
    preferred_duration: float = 3.5,
    boundary_predicate: BoundaryPredicate = _default_boundary,
    track_index: int = 1,
) -> list[RefinedSegment]:
    """Segment one source's whole word stream, ignoring raw segment edges.

    Transcription segments are an artifact of the recognizer, not of the
    speech: they can end in the middle of a word or bunsetsu. Merging the
    words of every raw segment of a source before segmentation keeps slice
    boundaries at bunsetsu boundaries across those edges too. Each refined
    segment keeps the ``no_speech_prob`` of the raw segment its first word
    came from.
    """
    raws = list(raw_segments)
    if len(raws) <= 1 or any(not raw.words for raw in raws):
        result: list[RefinedSegment] = []
        for raw in raws:
            result.extend(
                segment_raw(
                    raw,
                    max_characters,
                    max_duration,
                    preferred_duration,
                    boundary_predicate,
                    track_index,
                )
            )
        return result

    words: list[WordTimestamp] = []
    no_speech_by_word: list[float] = []
    for raw in raws:
        words.extend(raw.words)
        no_speech_by_word.extend([raw.no_speech_prob] * len(raw.words))
    merged = RawSegment(
        id=raws[0].id,
        start=min(raw.start for raw in raws),
        end=max(raw.end for raw in raws),
        text="",
        no_speech_prob=raws[0].no_speech_prob,
        words=words,
        source_id=raws[0].source_id,
    )
    refined = segment_raw(
        merged,
        max_characters,
        max_duration,
        preferred_duration,
        boundary_predicate,
        track_index,
    )
    word_index = 0
    for segment in refined:
        segment.no_speech_prob = no_speech_by_word[word_index]
        word_index += len(segment.words)
    return refined
