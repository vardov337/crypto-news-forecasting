# Воспроизводимое сравнение моделей прогнозирования криптовалют с учётом новостного фона

*A reproducible protocol for comparing hourly BTC and ETH return forecasting models (ARIMA/ARIMAX, Random Forest, XGBoost, LSTM) with English- and Russian-language news sentiment. All data are collected by scripts, all parameters live in one config file, and every table and figure of the paper is produced by the code in this repository.*

**Статус:** расчёты для статьи выполнены. Протокол эксперимента — [`PROTOCOL.md`](PROTOCOL.md): версия 1.0, утверждён до начала расчётов (01.10.2026); все последующие изменения с датой и причиной — в его разделе 12 «Отклонения от протокола».

## Структура

```
├── PROTOCOL.md            протокол эксперимента: решения фиксируются до расчётов
├── configs/config.yaml    все параметры: данные, признаки, валидация, модели, метрики
├── src/cryptonews/        пакет с кодом пайплайна
│   ├── config.py          загрузка конфигурации и путей
│   ├── utils.py           зёрна, логирование, хеши, манифест запуска
│   ├── data/              цены Binance, CryptoVision, русскоязычные новости
│   ├── sentiment.py       модели тональности: кандидаты, ревизии, расчёт вероятностей
│   ├── annotation.py      выборка для ручной разметки, файлы Excel, macro-F1 и каппа
│   ├── period.py          границы общей выборки
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
├── checks/                ручные проверки данных (сверка времени публикации новостей с сайтами)
├── data/                  данные (в git не хранятся, см. data/README.md)
└── results/               таблицы, рисунки, метрики, манифесты запусков
```

## Установка

Нужен Python 3.10 или новее.

```bash
git clone https://github.com/vardov337/crypto-news-forecasting.git
cd crypto-news-forecasting
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
| 5 | `05_sentiment.py` — границы выборки, файлы для ручной разметки, тональность всех заголовков моделями-кандидатами (нужна видеокарта) | 5.2–5.6 |
| 5b | `05b_choose_sentiment.py` — проверка моделей на ручной разметке, выбор модели по macro-F1 | 5.7 |
| 6 | `06_build_features.py` — выравнивание новостей по часам, признаки, схема walk-forward (`results/splits.json`) | 6.1–6.4, 7.1 |
| 7 | `07_run_models.py` — модели по схеме walk-forward по группам: `--family baselines`, `arima`, `trees`, `lstm`; `--smoke` — пробный прогон; `--models … --restart` — пересчитать отдельные модели заново (прежние прогнозы переносятся в `results/predictions_old/`) | 7.1–7.5 |
| 8 | `08_evaluate.py` — метрики, тесты Диболда–Мариано и Песарана–Тиммерманна, Model Confidence Set, бэктест с издержками, выбор модели | 8.1–8.4 |
| 8б | `08b_leakage_checks.py` — тест сдвига новостей и плацебо-тест (раздел 10 протокола) | 8.4 |
| 8в | `08c_granger.py` — тест Грейнджера в обе стороны: стационарность, лаги 1/6/24 ч и по BIC, HAC | 8.4 |
| 9 | `09_make_report.py` — таблицы статьи и приложения (`results/report/tables.xlsx`: Т1–Т5, П1–П12), рисунки статьи в её порядке (`fig1_scheme` … `fig6_shift`, PNG и SVG), точные версии пакетов (`requirements-lock.txt`); всё вместе — `results/report.zip` | 8.5 |

Что нужно для полного прогона:

- архив CryptoVision версии 2 положить вручную в `data/raw/cryptovision/` (см. [`data/README.md`](data/README.md));
- видеокарта для шагов 5 и 7 (в Colab — T4); остальные шаги работают на процессоре;
- время: сбор ForkLog — около часа, тональность — 10–20 минут, модели — несколько часов (шаг 7 можно прерывать: посчитанное не пересчитывается), остальные шаги — минуты.

Ручная разметка тональности уже лежит в `annotation/labels/`; шаг 5b берёт её оттуда.

Разовые проверки источников, на которых основан выбор сайтов и способа сбора, лежат в
`scripts/diagnostics/` и в основной прогон не входят.

Каждый шаг пишет манифест запуска (время, коммит, версии пакетов, хеши входов) в `results/`.

## Данные

Сырые данные в репозитории не хранятся: цены и новости скачиваются скриптами, источники и лицензии описаны в [`data/README.md`](data/README.md).

## Результаты

Таблицы и рисунки статьи строит шаг 9 в `results/report/` (архив `results/report.zip`): таблицы статьи Т1–Т5
и приложения П1–П12 — в `tables.xlsx`, рисунки — в PNG и SVG. Результаты, по которым написана статья (прогнозы
всех моделей, метрики, таблицы, манифесты запусков), архивируются вместе с релизом кода в Zenodo.

## Как цитировать

См. [`CITATION.cff`](CITATION.cff). DOI архива в Zenodo появится после релиза.

## Лицензия

Код — MIT (см. `LICENSE`). Данные — по лицензиям их источников.
