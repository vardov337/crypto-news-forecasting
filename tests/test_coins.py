"""Проверки привязки новостей к монетам. Примеры взяты в стиле настоящих заголовков."""
import pandas as pd

from cryptonews.data import coins

BTC_YES = [
    "Банк Citi ждет скорого роста биткоина до $113 000",
    "Биткойн обновил максимум",
    "Bitcoin ETF inflows hit record",
    "BTC drops below $60K",
    "Спотовые биткоин-ETF привлекли $1 млрд",
    "Bitcoin's hashrate climbs",
]
BTC_NO = [
    "Bitcoin Cash forks again",
    "Bitcoin SV miners leave",
    "WBTC supply on Ethereum grows",
    "Tether mints $1 billion USDT",
]
ETH_YES = [
    "Ethereum developers delay upgrade",
    "ETH staking yields fall",
    "Эфир подорожал на 5%",
    "Эфириум обогнал Solana по комиссиям",
    "Ether ETFs see outflows",
    "Курс эфира упал",
]
ETH_NO = [
    "Ethereum Classic hit by 51% attack",
    "В прямом эфире обсудили регулирование",
    "Tether mints $1 billion USDT",
    "Ripple wins case against SEC",
]


def test_btc_patterns():
    for title in BTC_YES:
        assert coins.mentions(title, "BTCUSDT"), title
    for title in BTC_NO:
        assert not coins.mentions(title, "BTCUSDT"), title


def test_eth_patterns():
    for title in ETH_YES:
        assert coins.mentions(title, "ETHUSDT"), title
    for title in ETH_NO:
        assert not coins.mentions(title, "ETHUSDT"), title


def test_tag_and_coverage_table():
    frame = pd.DataFrame({"title": ["BTC и ETH растут", "Биткоин падает", "Ripple выиграл суд", None]})
    tagged = coins.tag(frame)
    assert list(tagged["mentions_btc"]) == [True, True, False, False]
    assert list(tagged["mentions_eth"]) == [True, False, False, False]
    table = coins.coverage_table(tagged).set_index("Группа")["Новостей"]
    assert table["Упоминают обе монеты"] == 1 and table["Не упоминают ни одну из двух"] == 2
    assert table["Всего"] == 4
