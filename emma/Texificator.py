import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Iterable, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


TexificationType = Literal["full", "web"]

MAX_LENGTH = 150


class Texificator:
    """Представляет алгоритм тексификации текста."""
    
    def __init__(self):
        self._model_name = "basic-go/rut5-base-texificator"
        self._tokenizer = T5Tokenizer.from_pretrained(self._model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_name)
            
    @property
    def model_name(self) -> str:
        """Возвращает актуальное название модели, используемой для тексификации."""
        return self._model_name
    
    def _generate(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                eos_token_id=self._tokenizer.eos_token_id,
                max_length=MAX_LENGTH,
                num_beams=11,
                do_sample=True, 
                top_k=100, 
                early_stopping=True
            )
            
        return self._tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    def doit(self, items: Union[str, Iterable[str]], texification_type: TexificationType = "full") -> Union[str, Iterable[str]]:
        """
        Осуществляет тексификацию заданного типа.
        """
        if texification_type == "web":
            raise NotImplementedError("Веб-тексификация не реализована в текущей версии библиотеки.")
        elif texification_type not in ["full", "web"]:
            raise ValueError(f"Указан недопустимый тип тексификации `{texification_type}`.")
            
        if isinstance(items, str):
            return self._generate(items)
        elif isinstance(items, Iterable):
            return (self._generate(item) for item in items)
        raise TypeError("Параметр `items` имеет недопустимый тип.")