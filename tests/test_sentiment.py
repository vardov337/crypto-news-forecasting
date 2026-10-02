"""Тональность без настоящих моделей: сопоставление классов, пакеты, порядок, оценки.

Настоящие модели скачиваются только в шаге 5; здесь их заменяет функция, которая
по тексту возвращает заданные логиты, — этого достаточно, чтобы проверить, что
вероятности не перепутаны ни по классам, ни по строкам.
"""
import numpy as np
import pandas as pd
import pytest

from cryptonews import sentiment


def test_canonical_labels_by_name_not_position():
    assert sentiment.canonical_labels({0: "positive", 1: "negative", 2: "neutral"}) == \
        {0: "positive", 1: "negative", 2: "neutral"}
    assert sentiment.canonical_labels({"0": "Bearish", "1": "Neutral", "2": "Bullish"}) == \
        {0: "negative", 1: "neutral", 2: "positive"}
    assert sentiment.canonical_labels({0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"})[2] == "negative"
    assert sentiment.canonical_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                                      {"LABEL_0": "negative", "label_1": "neutral", "LABEL_2": "positive"}) == \
        {0: "negative", 1: "neutral", 2: "positive"}


def test_canonical_labels_rejects_unknown_or_incomplete():
    with pytest.raises(ValueError, match="LABEL_0"):
        sentiment.canonical_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValueError, match="ровно три"):
        sentiment.canonical_labels({0: "positive", 1: "negative"})


def test_column_order_and_softmax():
    mapping = sentiment.canonical_labels({0: "positive", 1: "negative", 2: "neutral"})
    assert sentiment.column_order(mapping) == [1, 2, 0]
    probs = sentiment.softmax(np.array([[1000.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1) and probs[0, 0] > 0.999 and np.allclose(probs[1], 1 / 3)


def test_batches_cover_every_text_once_sorted_by_length():
    lengths = [5, 1, 9, 3, 7, 2, 8]
    batches = sentiment.batches_by_length(lengths, 3)
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(len(lengths)))
    assert [lengths[i] for i in flat] == sorted(lengths)


def fake_logits(texts):
    """Модель с порядком классов (positive, negative, neutral), как у ProsusAI/finbert."""
    out = []
    for t in texts:
        if "up" in t:
            out.append([5.0, 0.0, 1.0])
        elif "down" in t:
            out.append([0.0, 5.0, 1.0])
        else:
            out.append([0.0, 0.0, 5.0])
    return np.array(out)


def test_predict_proba_keeps_rows_and_maps_classes():
    texts = ["market goes up a lot today", "down", "a very long neutral headline about nothing", "up"]
    mapping = sentiment.canonical_labels({0: "positive", 1: "negative", 2: "neutral"})
    seen = []
    probs = sentiment.predict_proba(texts, fake_logits, sentiment.column_order(mapping), batch_size=2,
                                    progress=lambda done, total: seen.append((done, total)))
    labels = np.asarray(sentiment.CLASSES)[probs.argmax(axis=1)]
    assert labels.tolist() == ["positive", "negative", "neutral", "positive"]
    assert seen[-1] == (4, 4)
    frame = sentiment.scores_frame(["u1", "u2", "u3", "u4"], probs)
    assert frame.columns.tolist() == sentiment.OUTPUT_COLUMNS
    assert frame.loc[0, "score"] > 0.9 and frame.loc[1, "score"] < -0.9 and abs(frame.loc[2, "score"]) < 0.05
    assert sentiment.label_shares(frame) == {"negative": 0.25, "neutral": 0.25, "positive": 0.5}


def test_sanity_check_counts_matches():
    def classify(texts):
        return ["positive" if ("high" in t or "rall" in t or "максимум" in t or "подорожал" in t)
                else "negative" for t in texts]
    correct, total, rows = sentiment.sanity_check("en", classify)
    assert (correct, total) == (4, 4)
    correct_ru, _, _ = sentiment.sanity_check("ru", classify)
    assert correct_ru == 4
    flipped, _, _ = sentiment.sanity_check("en", lambda texts: ["neutral"] * len(texts))
    assert flipped == 0 and rows[0]["expected"] == "positive"


def test_extremes_and_slug_and_candidates():
    news = pd.DataFrame({"url": ["a", "b", "c"], "title": ["good", "bad", "meh"]})
    scores = pd.DataFrame({"url": ["a", "b", "c"], "score": [0.9, -0.8, 0.0]})
    top, bottom = sentiment.extremes(news, scores, k=1)
    assert top == ["good"] and bottom == ["bad"]
    assert sentiment.model_slug("ProsusAI/finbert", "4556d13015211d73") == "ProsusAI__finbert@4556d130"
    cfg = {"sentiment": {"candidates": {"en": [{"model": "a/b", "revision": None}, "c/d"],
                                        "ru": [{"model": "e/f", "revision": "abc", "labels": {"x": "positive"}}]}}}
    cands = sentiment.candidates_from_config(cfg)
    assert [(c.language, c.name, c.revision) for c in cands] == [("en", "a/b", None), ("en", "c/d", None),
                                                                 ("ru", "e/f", "abc")]
    assert cands[2].labels == {"x": "positive"}
