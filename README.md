# Воспроизводимое сравнение моделей прогнозирования криптовалют с учётом новостного фона

*A reproducible protocol for comparing hourly BTC and ETH return forecasting models (ARIMA/ARIMAX, Random Forest, XGBoost, LSTM) with English- and Russian-language news sentiment. All data are collected by scripts, all parameters live in one config file, and every table and figure of the paper is produced by the code in this repository.*

**Статус:** работа в процессе. Протокол эксперимента — [`PROTOCOL.md`](PROTOCOL.md) (черновик, утверждается до начала расчётов).

## Структура

```
├── PROTOCOL.md            протокол эксперимента: решения фиксируются до расчётов
├── configs/config.yaml    все параметры: данные, признаки, валидация, модели, метрики
├── src/cryptonews/        пакет с кодом пайплайна
│   ├── config.py          загрузка конфигурации и путей
│   ├── utils.py           зёрна, логирование, хеши, манифест запуска
│   ├── data/              цены Binance, CryptoVision, русскоязычные новости
│   ├── sentiment.py       FinBERT и RuBERT
│   ├── align.py           выравнивание новостей по часовым свечам
│   ├── features.py        ценовые и новостные признаки
│   ├── validation.py      схема walk-forward
│   ├── models/            бенчмарки, ARIMA/ARIMAX, деревья, LSTM
│   ├── evaluation/        метрики, бэктест, статистические тесты
│   └── diagnostics.py     проверки на утечки и ошибки данных
├── scripts/               шаги пайплайна по порядку (00–09); diagnostics/ — разовые проверки
├── tests/                 модульные тесты
├── notebooks/             Colab-запуск и исследование; числа для статьи — только из scripts/
├── annotation/            инструкция и файлы ручной разметки тональности
├── data/                  данные (в git не хранятся, см. data/README.md)
└── results/               таблицы, рисунки, метрики, манифесты запусков
```

## Установка

Нужен Python 3.10 или новее.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
pip install -e .
python scripts/00_check_env.py      # проверка окружения и фиксация версий
```

### Google Colab

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vardov337/crypto-news-forecasting/blob/main/notebooks/colab.ipynb)

Ноутбук `notebooks/colab.ipynb` подключает Google Диск, скачивает свежую версию кода и запускает шаги
пайплайна. Данные и результаты лежат на Диске, пути к ним задают переменные `CRYPTONEWS_DATA_DIR`
и `CRYPTONEWS_RESULTS_DIR`, поэтому код и данные не смешиваются.

## Воспроизведение

```bash
bash run_all.sh                     # все шаги по порядку с configs/config.yaml
```

| Шаг | Скрипт | Задачи трекера |
| --- | --- | --- |
| 0 | `00_check_env.py` — окружение и версии | 0.6, 10.1 |
| 1 | `01_download_prices.py` — часовые свечи BTCUSDT и ETHUSDT (архивы data.binance.vision) | 3.1–3.2 |
| 2 | `02_collect_forklog.py` — русскоязычные новости ForkLog (программный интерфейс сайта) | 2.1–2.5 |
| 3 | `03_prepare_news_en.py` — CryptoVision: часовой пояс по сверке с Binance, UTC, дубли | 3.3–3.6 |
| 4 | `04_prepare_news_ru.py` — очистка русскоязычных новостей, привязка к монетам | 4.1–4.4 |
| 5 | `05_sentiment.py` — тональность | 5.5–5.7 |
| 6 | `06_build_features.py` — выравнивание и признаки | 6.1–6.4 |
| 7 | `07_run_models.py` — модели по схеме walk-forward | 7.1–7.5 |
| 8 | `08_evaluate.py` — метрики, бэктест, тесты | 8.1–8.4 |
| 9 | `09_make_report.py` — таблицы и рисунки статьи | 8.5 |

Разовые проверки источников, на которых основан выбор сайтов и способа сбора, лежат в
`scripts/diagnostics/` и в основной прогон не входят.

Каждый шаг пишет манифест запуска (время, коммит, версии пакетов, хеши входов) в `results/`.

## Данные

Сырые данные в репозитории не хранятся: цены и новости скачиваются скриптами, источники и лицензии описаны в [`data/README.md`](data/README.md).

## Как цитировать

См. [`CITATION.cff`](CITATION.cff). DOI архива в Zenodo появится после релиза.

## Лицензия

Код — MIT (см. `LICENSE`). Данные — по лицензиям их источников.
