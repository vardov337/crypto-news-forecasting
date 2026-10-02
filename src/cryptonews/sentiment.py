"""Тональность заголовков (задачи 5.5–5.7, PROTOCOL.md, раздел 5).

Для каждого языка оцениваются несколько моделей-кандидатов с Hugging Face. Каждая
модель загружается с зафиксированной ревизией — хешем коммита, поэтому повторный
запуск через год даст те же числа. Для каждой новости сохраняются вероятности трёх
классов, класс с максимальной вероятностью и оценка s = P(позитив) − P(негатив).

Какая модель пойдёт в признаки, решает ручная разметка (шаг 5b): выбирается
кандидат с наибольшим macro-F1. Правило зафиксировано до разметки.

Классы у моделей названы по-разному (positive, POSITIVE, Bullish …) и стоят в
разном порядке, поэтому сопоставление идёт по названию класса, а не по номеру.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

CLASSES = ("negative", "neutral", "positive")      # порядок колонок вероятностей
PROB_COLUMNS = ("p_neg", "p_neu", "p_pos")
OUTPUT_COLUMNS = ["url", *PROB_COLUMNS, "score", "label"]

SYNONYMS = {
    "negative": "negative", "neg": "negative", "bearish": "negative",
    "neutral": "neutral", "neu": "neutral",
    "positive": "positive", "pos": "positive", "bullish": "positive",
}

# Заведомо однозначные фразы: грубая проверка, что классы сопоставлены верно.
# Если модель путает рост с обвалом, перепутано сопоставление, а не «модель плохая».
SANITY = {
    "en": [
        ("Bitcoin surges to a new all-time high as institutional demand soars", "positive"),
        ("Ethereum rallies after a successful network upgrade", "positive"),
        ("Major crypto exchange hacked, $200 million in user funds stolen", "negative"),
        ("Regulators sue crypto lender for fraud as its token collapses", "negative"),
    ],
    "ru": [
        ("Биткоин обновил исторический максимум на фоне притока институциональных инвесторов", "positive"),
        ("Эфириум подорожал после успешного обновления сети", "positive"),
        ("Хакеры взломали криптобиржу и похитили $200 млн пользователей", "negative"),
        ("Регулятор подал в суд на криптокредитора, его токен обвалился", "negative"),
    ],
}


@dataclass
class Candidate:
    language: str
    name: str
    revision: str | None = None      # хеш коммита; None — взять текущий и записать
    labels: dict | None = None       # явное соответствие классов, если названия нестандартные


def candidates_from_config(cfg: dict) -> list[Candidate]:
    result = []
    for language, items in cfg["sentiment"]["candidates"].items():
        for item in items:
            item = {"model": item} if isinstance(item, str) else item
            result.append(Candidate(language=language, name=item["model"],
                                    revision=item.get("revision"), labels=item.get("labels")))
    return result


def model_slug(name: str, revision: str) -> str:
    """Имя файла с оценками модели: автор__модель@первые-8-знаков-ревизии."""
    return f"{name.replace('/', '__')}@{revision[:8]}"


def canonical_labels(id2label: dict, overrides: dict | None = None) -> dict[int, str]:
    """Номер класса модели → negative / neutral / positive."""
    overrides = {str(k).strip().lower(): v for k, v in (overrides or {}).items()}
    mapping: dict[int, str] = {}
    for index, name in id2label.items():
        key = str(name).strip().lower()
        canon = overrides.get(key) or SYNONYMS.get(key)
        if canon not in CLASSES:
            raise ValueError(f"Класс «{name}» не удаётся сопоставить с negative/neutral/positive; "
                             "укажите соответствие в конфигурации (поле labels у модели)")
        mapping[int(index)] = canon
    if sorted(mapping.values()) != sorted(CLASSES):
        raise ValueError(f"Нужны ровно три класса negative/neutral/positive, а получилось {mapping}")
    return mapping


def column_order(mapping: dict[int, str]) -> list[int]:
    """Номера классов модели в порядке CLASSES."""
    inverse = {canon: index for index, canon in mapping.items()}
    return [inverse[c] for c in CLASSES]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def batches_by_length(lengths: Sequence[int], batch_size: int) -> list[np.ndarray]:
    """Пакеты из текстов близкой длины: меньше пустого заполнения — быстрее расчёт."""
    order = np.argsort(np.asarray(lengths), kind="stable")
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def predict_proba(texts: Sequence[str], logits_fn: Callable[[list[str]], np.ndarray],
                  order: list[int], batch_size: int, lengths: Sequence[int] | None = None,
                  progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Вероятности классов (n × 3, порядок CLASSES) для каждого текста.

    logits_fn получает список текстов и возвращает логиты модели в её собственном
    порядке классов; order переставляет их в порядок CLASSES.
    """
    texts = list(texts)
    if lengths is None:
        lengths = [len(t) for t in texts]
    out = np.empty((len(texts), len(CLASSES)), dtype=np.float32)
    done = 0
    for idx in batches_by_length(lengths, batch_size):
        logits = logits_fn([texts[i] for i in idx])
        out[idx] = softmax(logits)[:, order].astype(np.float32)
        done += len(idx)
        if progress is not None:
            progress(done, len(texts))
    return out


def scores_frame(urls: Sequence[str], probs: np.ndarray) -> pd.DataFrame:
    """Таблица оценок: адрес новости, вероятности, s = P(позитив) − P(негатив), класс."""
    frame = pd.DataFrame(probs, columns=list(PROB_COLUMNS))
    frame.insert(0, "url", list(urls))
    frame["score"] = (frame["p_pos"] - frame["p_neg"]).astype(np.float32)
    frame["label"] = np.asarray(CLASSES)[np.asarray(probs).argmax(axis=1)]
    return frame[OUTPUT_COLUMNS]


def label_shares(frame: pd.DataFrame) -> dict[str, float]:
    shares = frame["label"].value_counts(normalize=True)
    return {c: round(float(shares.get(c, 0.0)), 4) for c in CLASSES}


def sanity_check(language: str, classify: Callable[[list[str]], list[str]]) -> tuple[int, int, list[dict]]:
    """Сколько однозначных фраз из SANITY модель отнесла к ожидаемому классу."""
    items = SANITY[language]
    predicted = classify([text for text, _ in items])
    rows = [{"text": t, "expected": e, "predicted": p} for (t, e), p in zip(items, predicted)]
    correct = sum(r["expected"] == r["predicted"] for r in rows)
    return correct, len(items), rows


def extremes(news: pd.DataFrame, scores: pd.DataFrame, k: int = 3) -> tuple[list[str], list[str]]:
    """Заголовки с самой высокой и самой низкой оценкой — для глазной проверки."""
    merged = news[["url", "title"]].merge(scores[["url", "score"]], on="url")
    ordered = merged.sort_values(["score", "url"])
    return list(ordered["title"].tail(k)[::-1]), list(ordered["title"].head(k))


# --- Работа с моделями Hugging Face. torch и transformers импортируются здесь,
# --- чтобы остальной пакет работал и без них.

def resolve_revision(name: str, revision: str | None = None) -> str:
    """Полный хеш коммита модели: заданной ревизии или текущей ветки main."""
    from huggingface_hub import HfApi

    return HfApi().model_info(name, revision=revision or "main").sha


def pick_device(allow_cpu: bool = False) -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise SystemExit(
        "Видеокарта не найдена. В Colab: меню «Среда выполнения» → «Сменить среду выполнения» → "
        "«Графический процессор T4» → «Сохранить», затем снова ячейки 1–3 и этот шаг. "
        "Без видеокарты расчёт займёт пару часов; если это осознанно, добавьте ключ --cpu."
    )


def device_name(device: str) -> str:
    import torch

    return torch.cuda.get_device_name(0) if device == "cuda" else "CPU"


class HFClassifier:
    """Классификатор тональности с Hugging Face на заданной ревизии."""

    def __init__(self, name: str, revision: str, device: str, max_length: int,
                 label_overrides: dict | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.revision, self.device, self.max_length = name, revision, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
        self.model.float()      # одинаковая точность для всех моделей, как бы ни были сохранены веса
        self.model.to(device)
        self.model.eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.mapping = canonical_labels(self.id2label, label_overrides)
        self.order = column_order(self.mapping)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))

    def token_lengths(self, texts: Sequence[str], chunk: int = 2048) -> np.ndarray:
        lengths: list[int] = []
        for i in range(0, len(texts), chunk):
            encoded = self.tokenizer(list(texts[i:i + chunk]), add_special_tokens=True,
                                     truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return np.asarray(lengths)

    def logits(self, batch: list[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            return self.model(**encoded).logits.float().cpu().numpy()

    def classify(self, texts: list[str]) -> list[str]:
        probs = predict_proba(texts, self.logits, self.order, batch_size=len(texts) or 1)
        return list(np.asarray(CLASSES)[probs.argmax(axis=1)])

    def close(self) -> None:
        del self.model
        free_memory()


def free_memory() -> None:
    """Освобождает память видеокарты после модели."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
