import re
import string
import unicodedata
from typing import Optional

from transformers import pipeline
from transliterate import translit

space_expr = re.compile(r"\s+")


def _preprocess(s: str) -> str:
    s = space_expr.sub(" ", s)
    s = translit(s, "ru", reversed=True)
    r = (
        set(s)
        - set(string.punctuation)
        - set(string.ascii_letters)
        - set(string.digits)
        - set(string.whitespace)
    )
    for char in r:
        try:
            s = s.replace(char, " " + unicodedata.name(char) + " ")
        except:
            s = s.replace(char, " [UNK] ")
    return space_expr.sub(" ", s).strip()


class Moderator:
    """Представляет алгоритм модерации текста с математическими сущностями."""

    def __init__(
        self, device: Optional[int] = None, model: str = "basic-go/texbert-moderator"
    ):
        self._model = pipeline("text-classification", model=model, device=device)

    def moderate(self, input_text: str) -> bool:
        """Возвращает значение, указывающее проходит ли входной текст модерацию."""
        prediction = self.moderate_proba(input_text)
        return prediction[0]["label"] == "tex"

    def moderate_proba(self, input_text: str) -> bool:
        """Возвращает метку входного текста с оценкой уверенности классификации."""
        input_text = _preprocess(input_text)
        return self._model(input_text)
