# emma

Emma — это библиотека для обработки образовательного математического контента на русском языке.

Библиотека позволяет решать следующие задачи:
- [тексификация математических выражений](notebooks/texification.ipynb);
- [автомодерация математического контента](notebooks/moderation.ipynb);
- [получение векторных представлений текстов с математическими выражениями](notebooks/embeddings.ipynb). 

## Минимальные технические требования

- CPU с 2 ядрами;
- GPU с 8 Гб VRAM;
- 8 Гб RAM;
- Python >= 3.10;
- CUDA >= 11.7.

## Установка

Для установки необходимо выполнить следующие шаги:

1. клонировать репозиторий библиотеки:

```sh
git clone https://github.com/basic-go-ahead/emma
```

2. выполнить установку пакета в выбранном окружении:

```sh
pip install ./emma --extra-index-url https://download.pytorch.org/whl/cu117
```