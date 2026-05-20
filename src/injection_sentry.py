"""Injection Sentry — 3-way weighted ensemble for prompt injection detection."""

from __future__ import annotations

import html as html_mod
import re
import unicodedata
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class InjectionSentryEnsemble:
    """Weighted ensemble of XLM-RoBERTa + 2x DeBERTa-v3 for prompt injection detection.

    Returns a single boolean from ``evaluate(text)``: ``True`` when the weighted
    softmax score for the injection class meets or exceeds ``THRESHOLD``.
    """

    model_name = "Injection Sentry"
    WEIGHTS: tuple[float, float, float] = (0.36, 0.26, 0.38)
    THRESHOLD: float = 0.57
    REPOS: tuple[str, str, str] = (
        "Verm1ion/injection-sentry-xlmr",
        "Verm1ion/injection-sentry-deberta",
        "Verm1ion/injection-sentry-deberta-v2",
    )

    ZERO_WIDTH = frozenset("​‌‍⁠﻿­‎‏")
    UNICODE_TAG_RE = re.compile(r"[\U000E0000-\U000E007F]")
    HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    WHITESPACE_RE = re.compile(r"\s+")
    INJECTION_LABEL_HINTS = ("INJ", "UNSAFE", "MALICIOUS", "ATTACK")

    def __init__(self, repos: Iterable[str] | None = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.components: list[tuple[AutoTokenizer, AutoModelForSequenceClassification, int]] = []
        for repo in (repos if repos is not None else self.REPOS):
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = (
                AutoModelForSequenceClassification.from_pretrained(repo)
                .to(self.device)
                .eval()
            )
            self.components.append((tokenizer, model, self._injection_index(model)))

    @classmethod
    def _injection_index(cls, model) -> int:
        id2label = model.config.id2label or {}
        if isinstance(id2label, list):
            id2label = dict(enumerate(id2label))
        for index, label in id2label.items():
            if any(hint in str(label).upper() for hint in cls.INJECTION_LABEL_HINTS):
                return int(index)
        return 1

    @classmethod
    def _preprocess(cls, text: str) -> str:
        text = html_mod.unescape(str(text))
        text = cls.UNICODE_TAG_RE.sub(" ", text)
        text = cls.HTML_COMMENT_RE.sub(" ", text)
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ch not in cls.ZERO_WIDTH)
        return cls.WHITESPACE_RE.sub(" ", text).strip()

    def _score_one(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForSequenceClassification,
        inj_idx: int,
        text: str,
        stride: int = 128,
    ) -> float:
        text = self._preprocess(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= 510:
            encoded = tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = model(**encoded).logits.float()
            return torch.softmax(logits, dim=-1)[0, inj_idx].item()

        step = 510 - stride
        chunk_scores: list[float] = []
        for start in range(0, len(token_ids), step):
            chunk = tokenizer.decode(token_ids[start : start + 510])
            encoded = tokenizer(
                chunk, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = model(**encoded).logits.float()
            chunk_scores.append(torch.softmax(logits, dim=-1)[0, inj_idx].item())
            if start + 510 >= len(token_ids):
                break
        return max(chunk_scores)

    def score(self, text: str) -> float:
        """Return the raw weighted ensemble score in ``[0, 1]``."""
        return sum(
            weight * self._score_one(tokenizer, model, inj_idx, text)
            for weight, (tokenizer, model, inj_idx) in zip(self.WEIGHTS, self.components)
        )

    def evaluate(self, text: str) -> bool:
        """Return ``True`` if ``text`` is classified as a prompt injection."""
        return self.score(text) >= self.THRESHOLD
