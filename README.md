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
├── scripts/               шаги пайплайна по порядку (00–09)
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

Откройте `notebooks/00_colab_setup.ipynb`: он подключает Google Drive для данных, клонирует репозиторий, ставит пакет и проверяет окружение. Данные хранятся на Drive, путь к ним задаётся переменной `CRYPTONEWS_DATA_DIR`, поэтому код и данные не смешиваются.

## Воспроизведение

```bash
bash run_all.sh                     # все шаги по порядку с configs/config.yaml
```

| Шаг | Скрипт | Задачи трекера |
| --- | --- | --- |
| 0 | `00_check_env.py` — окружение и версии | 0.6, 10.1 |
| 1 | `01_download_prices.py` — свечи Binance | 3.1–3.2 |
| 1а | `01a_probe_ru_sites.py` — разведка русскоязычных сайтов | 1.1–1.4 |
| 1б | `01b_check_archive_depth.py` — глубина архива источников | 1.5 |
| 1в | `01c_probe_bulk_access.py` — способы массового сбора новостей | 2.1–2.3 |
| 2 | `02_prepare_news_en.py` — CryptoVision | 3.3–3.6 |
| 3 | `02_collect_forklog.py` — сбор русскоязычных новостей ForkLog | 2.1–2.5 |
| 4 | `04_prepare_news_ru.py` — очистка русскоязычных новостей | 4.1–4.4 |
| 5 | `05_sentiment.py` — тональность | 5.5–5.7 |
| 6 | `06_build_features.py` — выравнивание и признаки | 6.1–6.4 |
| 7 | `07_run_models.py` — модели по схеме walk-forward | 7.1–7.5 |
| 8 | `08_evaluate.py` — метрики, бэктест, тесты | 8.1–8.4 |
| 9 | `09_make_report.py` — таблицы и рисунки статьи | 8.5 |

Каждый шаг пишет манифест запуска (время, коммит, версии пакетов, хеши входов) в `results/`.

## Данные

Сырые данные в репозитории не хранятся: цены и новости скачиваются скриптами, источники и лицензии описаны в [`data/README.md`](data/README.md).

## Как цитировать

См. [`CITATION.cff`](CITATION.cff). DOI архива в Zenodo появится после релиза.

## Лицензия

Код — MIT (см. `LICENSE`). Данные — по лицензиям их источников.
