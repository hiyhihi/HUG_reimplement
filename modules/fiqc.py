"""Deterministic query corruptions for the Fashion-IQ-C benchmark.

The benchmark is manifest-backed: source Fashion-IQ images are never copied or
modified.  A row plus ``seed`` and ``query_index`` is sufficient to reproduce
the corrupted query exactly.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageFilter


FIQC_SCHEMA_VERSION = 2
FIQC_SUITE = (
    ("image", "gaussian_blur"),
    ("image", "occlusion"),
    ("text", "word_deletion"),
    ("text", "spelling_error"),
    ("text", "synonym_replacement"),
    ("text", "semantic_contradiction"),
)

_SYNONYMS = {
    "big": "large", "large": "big", "small": "little", "little": "small",
    "long": "lengthy", "short": "cropped", "dark": "deep", "light": "pale",
    "shirt": "top", "top": "shirt", "dress": "gown", "gown": "dress",
    "sleeve": "arm", "sleeves": "arms", "pattern": "print", "patterns": "prints",
    "striped": "stripe-patterned", "plain": "solid", "loose": "relaxed",
    "tight": "fitted", "formal": "dressy", "casual": "informal",
    "brighter": "more vivid", "darker": "deeper", "longer": "more lengthy",
    "shorter": "more cropped", "more": "extra", "less": "reduced",

    # Fashion-IQ captions use these garment constructions frequently. The
    # replacements stay lexical/semantic-preserving; colours are deliberately
    # not remapped to nearby shades because that would change the query target.
    "has": "features", "with": "featuring", "solid": "plain",
    "color": "shade", "colored": "hued", "straps": "shoulder-straps",
    "strap": "shoulder-strap", "sleeveless": "without-sleeves",
    "strapless": "without-straps", "lighter": "paler", "print": "pattern",
    "printed": "patterned", "floral": "flower-patterned", "neck": "neckline",
    "neckline": "neck", "belt": "waist-belt", "belted": "waist-belted",
    "shiny": "glossy", "tighter": "more-fitted", "looser": "more-relaxed",
}

_ANTONYMS = {
    "black": "white", "white": "black", "dark": "light", "light": "dark",
    "long": "short", "short": "long", "longer": "shorter", "shorter": "longer",
    "large": "small", "big": "small", "small": "large", "loose": "tight",
    "tight": "loose", "fitted": "baggy", "baggy": "fitted", "plain": "patterned",
    "patterned": "plain", "striped": "plain", "bright": "dark", "brighter": "darker",
    "darker": "brighter", "formal": "casual", "casual": "formal", "with": "without",
    "has": "lacks", "have": "lack", "adds": "removes", "add": "remove",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[^A-Za-z]+")


def _stable_rng(seed: int, query_index: int, name: str) -> random.Random:
    digest = hashlib.sha256(
        f"fiqc-v{FIQC_SCHEMA_VERSION}|{seed}|{query_index}|{name}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _replace_word(token: str, replacement: str) -> str:
    if token[:1].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return replacement


class FIQCCorruptionGenerator:
    """Generate nested severity 0..4 corruptions with per-query stable RNG."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def image(
        self, image: Image.Image, name: str, severity: int, query_index: int,
    ) -> Tuple[Image.Image, Dict]:
        self._validate(name, severity, "image")
        if severity == 0:
            return image.copy(), {"type": "identity", "changed": False}
        if name == "gaussian_blur":
            radius = (0.0, 0.5, 1.0, 2.0, 4.0)[severity]
            return image.filter(ImageFilter.GaussianBlur(radius)), {
                "type": name, "radius": radius, "changed": True,
            }

        rng = _stable_rng(self.seed, query_index, name)
        width, height = image.size
        ratio = (0.0, 0.10, 0.20, 0.35, 0.50)[severity]
        side = math.sqrt(ratio)
        block_w, block_h = max(1, round(width * side)), max(1, round(height * side))
        # A fixed normalized centre makes higher severities spatially nested.
        center_x, center_y = rng.random() * width, rng.random() * height
        left = min(max(0, round(center_x - block_w / 2)), width - block_w)
        top = min(max(0, round(center_y - block_h / 2)), height - block_h)
        result = image.copy()
        result.paste((127, 127, 127), (left, top, left + block_w, top + block_h))
        return result, {
            "type": name, "area_ratio": ratio, "box": [left, top, block_w, block_h],
            "changed": True,
        }

    def text(self, text: str, name: str, severity: int, query_index: int) -> Tuple[str, Dict]:
        self._validate(name, severity, "text")
        if severity == 0:
            return text, {"type": "identity", "changed": False}
        if name == "word_deletion":
            return self._word_deletion(text, severity, query_index)
        if name == "spelling_error":
            return self._spelling_error(text, severity, query_index)
        if name == "synonym_replacement":
            return self._lexical_replacement(text, severity, query_index, _SYNONYMS, False)
        return self._lexical_replacement(text, severity, query_index, _ANTONYMS, True)

    @staticmethod
    def _validate(name: str, severity: int, modality: str) -> None:
        valid = {item_name for item_modality, item_name in FIQC_SUITE if item_modality == modality}
        if name not in valid:
            raise ValueError(f"invalid Fashion-IQ-C {modality} corruption: {name}")
        if severity not in range(5):
            raise ValueError("severity must be in [0, 4]")

    def _word_deletion(self, text: str, severity: int, query_index: int) -> Tuple[str, Dict]:
        words = text.split()
        if len(words) <= 1:
            return text, {"type": "word_deletion", "rate": 0.0, "changed": False}
        rng = _stable_rng(self.seed, query_index, "word_deletion")
        rate = (0.0, 0.10, 0.20, 0.35, 0.50)[severity]
        count = min(len(words) - 1, max(1, round(rate * len(words))))
        priority = sorted((rng.random(), index) for index in range(len(words)))
        removed = {index for _, index in priority[:count]}
        result = " ".join(word for index, word in enumerate(words) if index not in removed)
        return result, {
            "type": "word_deletion", "rate": rate, "removed_indices": sorted(removed),
            "changed": result != text,
        }

    def _spelling_error(self, text: str, severity: int, query_index: int) -> Tuple[str, Dict]:
        chars = list(text)
        candidates = [
            index for index in range(len(chars) - 1)
            if chars[index].isalpha() and chars[index + 1].isalpha()
        ]
        if not candidates:
            return text, {"type": "spelling_error", "rate": 0.0, "changed": False}
        rng = _stable_rng(self.seed, query_index, "spelling_error")
        rate = (0.0, 0.05, 0.10, 0.20, 0.30)[severity]
        priority = sorted((rng.random(), index) for index in candidates)
        count = min(len(priority), max(1, round(rate * len(priority))))
        selected = sorted((index for _, index in priority[:count]), reverse=True)
        # Adjacent selected positions may overlap; skip a second edit of the same pair.
        used = set()
        applied = []
        for index in selected:
            if index in used or index + 1 in used:
                continue
            chars[index], chars[index + 1] = chars[index + 1], chars[index]
            used.update((index, index + 1)); applied.append(index)
        result = "".join(chars)
        return result, {
            "type": "spelling_error", "rate": rate, "transpositions": sorted(applied),
            "changed": result != text,
        }

    def _lexical_replacement(
        self, text: str, severity: int, query_index: int, mapping: Dict[str, str],
        contradiction: bool,
    ) -> Tuple[str, Dict]:
        pieces = _TOKEN_RE.findall(text)
        candidates = [index for index, token in enumerate(pieces) if token.lower() in mapping]
        name = "semantic_contradiction" if contradiction else "synonym_replacement"
        rng = _stable_rng(self.seed, query_index, name)
        priority = sorted((rng.random(), index) for index in candidates)
        fraction = (0.0, 0.25, 0.50, 0.75, 1.0)[severity]
        count = min(len(priority), max(1, math.ceil(fraction * len(priority)))) if priority else 0
        replaced = []
        for _, index in priority[:count]:
            original = pieces[index]
            pieces[index] = _replace_word(original, mapping[original.lower()])
            replaced.append({"index": index, "from": original, "to": pieces[index]})
        result = "".join(pieces)
        if contradiction and not replaced:
            # Fashion captions occasionally contain no known attribute.  The
            # explicit negation keeps the corruption defined and auditable.
            result = f"not ({text})"
        return result, {
            "type": name, "fraction": fraction, "replacements": replaced,
            "fallback_negation": contradiction and not bool(replaced),
            "changed": result != text,
        }


def make_query_id(category: str, candidate_id: str, target_id: str, query_index: int) -> str:
    """Stable identity independent of corruption condition."""
    return f"fashion-iq:{category}:{query_index}:{candidate_id}->{target_id}"


def audit_text_corruptions(
    texts: Iterable[str], seed: int = 42, severities: Iterable[int] = range(1, 5),
) -> List[Dict]:
    """Return changed-rate diagnostics without running retrieval.

    This is intentionally cheap enough for preflight. It prevents a long label
    build from silently producing a lexical corruption with inadequate coverage.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("cannot audit an empty caption collection")
    generator = FIQCCorruptionGenerator(seed)
    output = []
    for modality, name in FIQC_SUITE:
        if modality != "text":
            continue
        for severity in severities:
            changed = 0
            for query_index, text in enumerate(texts):
                _, metadata = generator.text(text, name, severity, query_index)
                changed += int(metadata["changed"])
            output.append({
                "modality": modality,
                "corruption_type": name,
                "severity": severity,
                "count": len(texts),
                "changed_rate": changed / len(texts),
            })
    return output
