"""Phase 5 — Claim extraction (SRS FR-015, §8).

Splits a generated answer into verify-able claims. A claim is a sentence or
bullet item that asserts something policy-related; heading lines, citation-only
fragments and empty segments are skipped. ``[n]`` markers are pulled out of the
claim text so the validator can locate the evidence each claim pointed at.

Conclusions (the model's own inference — explicitly introduced by a
"Conclusion"/"Summary"/"Note" heading, as the Phase 4 comparison prompt asks)
are extracted too but labelled ``kind="conclusion"`` so the validator can treat
them differently from facts, matching FR-012's fact/conclusion distinction.
"""

from __future__ import annotations

import re
from typing import Sequence

from .schema import Claim

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

# A markdown-style heading line: optional bold markers, ends with a colon,
# and carries no sentence punctuation at the end beyond the colon.
_HEADING_RE = re.compile(r"^\s*\**\s*[A-Za-z][A-Za-z /-]{2,48}:\**\s*$")

# Headings that switch extracted claims from facts to conclusions.
_CONCLUSION_HEADINGS = ("conclusion", "summary", "recommendation", "verdict", "caveat", "note")

# Sentence split: after . ! ? followed by whitespace, but not after a
# citation marker or an abbreviation dot we cannot distinguish anyway.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MIN_CLAIM_CHARS = 12


class ClaimExtractor:
    """Extracts verify-able claims from a generated answer."""

    def __init__(self, min_chars: int = MIN_CLAIM_CHARS) -> None:
        self.min_chars = min_chars

    # ------------------------------------------------------------------

    def extract(self, answer: str) -> list[Claim]:
        """Split ``answer`` into claims.

        Punctuation/formatting the model added for its own benefit is not
        part of what a claim asserts, so the returned ``claim.text`` keeps the
        original sentence minus the citation markers.
        """
        claims: list[Claim] = []
        index = 0
        section = ""

        for line_no, raw_line in enumerate((answer or "").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            heading = self._heading_text(line)
            if heading is not None:
                section = heading.lower()
                continue
            kind = "conclusion" if any(h in section for h in _CONCLUSION_HEADINGS) else "fact"

            # Drop list prefixes ("- ", "* ", "1. ") — they are format, not content.
            content = re.sub(r"^[-*•]\s*", "", re.sub(r"^\d+\.\s+", "", line))

            segments = self._segments(content)
            # Models often place the citation right after the sentence period
            # ("... here. [1]"), which would orphan "[1]" into its own dropped
            # segment. Merge citation-only segments back into their neighbour.
            merged: list[tuple[str, list[int]]] = []
            for segment in segments:
                seg_markers = [int(m.group(1)) for m in _MARKER_RE.finditer(segment)]
                seg_text = _MARKER_RE.sub("", segment).strip(" *").strip()
                if seg_text or not merged:
                    merged.append((segment, seg_markers))
                else:
                    prev_text, prev_markers = merged[-1]
                    merged[-1] = (prev_text, prev_markers + seg_markers)

            for sentence, seg_markers in merged:
                text = _MARKER_RE.sub("", sentence).strip(" *").strip()
                markers = list(dict.fromkeys(seg_markers))
                if len(text) < self.min_chars or not any(ch.isalpha() for ch in text):
                    continue
                index += 1
                claims.append(
                    Claim(
                        text=text,
                        markers=markers,
                        kind=kind,
                        section=section,
                        index=index,
                        line=line_no,
                    )
                )
        return claims

    # ------------------------------------------------------------------

    @staticmethod
    def _heading_text(line: str) -> str | None:
        """Return the heading text when ``line`` is a heading, else None."""
        if not _HEADING_RE.match(line):
            return None
        text = line.strip(" *").rstrip(":").strip()
        return text or None

    @staticmethod
    def _segments(content: str) -> Sequence[str]:
        """Split a line into sentences, keeping bullets as one segment."""
        if content.startswith(("-", "*", "•")):
            return [content]
        return _SENTENCE_SPLIT_RE.split(content)
