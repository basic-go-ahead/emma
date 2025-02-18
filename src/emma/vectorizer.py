"""Определяет необходимую для векторизации текстов функциональность."""

import re
import string
import unicodedata
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

_space_expr = re.compile(r"\s+")
_cyrillic_letters = set(
    "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
)


def _preprocess(s: str) -> str:
    s = _space_expr.sub(" ", s)
    r = (
        set(s)
        - set(string.punctuation)
        - set(string.ascii_letters)
        - set(string.digits)
        - set(string.whitespace)
        - _cyrillic_letters
    )
    for char in r:
        try:
            s = s.replace(char, " " + unicodedata.name(char) + " ")
        except:  # pylint: disable=bare-except
            s = s.replace(char, " [UNK] ")
    return _space_expr.sub(" ", s).strip()


class Vectorizer(SentenceTransformer):  # type: ignore[misc]
    """Представляет векторизующую модель для текстом с математическими выражениями."""

    def encode(  # type: ignore[no-untyped-def]
        self, sentences: str | List[str], **kwargs: Dict[str, Any]
    ):  # pylint: disable=arguments-differ
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = [_preprocess(s) for s in sentences]
        return super().encode(sentences, **kwargs)
