import re
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from typing import Any, Dict, Iterable, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


TexificationType = Literal["full", "web"]

MAX_LENGTH = 128

class Texificator:
    """Представляет алгоритм тексификации текста."""

    def __init__(self, device: Optional[str] = None, model: Optional[str] = None):
        if device is None:
          device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
          model = "basic-go/rut5-base-texificator-st1"

        self.corrector = pipeline("text2text-generation", model="ai-forever/sage-fredt5-distilled-95m")

        self.device = device
        self._model_name = model
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_name)
        self._model.to(device)
        self._model.eval()
        self._params = dict(eos_token_id=self._tokenizer.eos_token_id,
            max_length=128,
            num_beams=7,
            do_sample=True,
            top_k=100,
            early_stopping=True,
            repetition_penalty=2.5, length_penalty=0.5,
        )

    def _retrieve(self, v: Union[str, Iterable[str]]):
        if isinstance(v, str):
            return self._correct(v)
        elif isinstance(v, Iterable):
            return (self._correct(e) for e in v)

    def _correct(self, text: str):
      if re.finditer("[А-Яа-яЁё]+", text):
        f = [x.group() for x in re.finditer(r"\\\(([^А-Яа-яЁё]+)\\\)", text)] + [x.group() for x in re.finditer(r"\\\[(.+)\\\]", text)]
        for k, e in enumerate(f):
          text = text.replace(e, f"[{k}] ")
        text = self.corrector(text)[0]["generated_text"]
        for k, e in enumerate(f):
          text = text.replace(f"[{k}]", e)
      return text

    def set_generation_params(self, params: Dict[str, Any]):
      self._params = params

    @property
    def model_name(self) -> str:
        """Возвращает название модели, используемой для тексификации."""
        return self._model_name

    def _generate(self, text: str) -> str:
        inputs = { k: v.to(self.device) for k, v in self._tokenizer(text, return_tensors="pt").items() }

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **self._params)

        return self._tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def _make_prompt(self, query: str, texification_type: TexificationType) -> str:
        return query if texification_type != "web" else "web: " + query

    def doit(self, items: Union[str, Iterable[str]], texification_type: TexificationType = "full") -> Union[str, Iterable[str]]:
        """
        Осуществляет тексификацию заданного типа.
        """
        if texification_type not in ["full", "web"]:
            raise ValueError(f"Указан недопустимый тип тексификации `{texification_type}`.")

        if isinstance(items, str):
            return self._retrieve(self._generate(self._make_prompt(items, texification_type)))
        elif isinstance(items, Iterable):
            return self._retrieve((self._make_prompt(self._generate(item), texification_type) for item in items))
        raise TypeError("Параметр `items` имеет недопустимый тип.")