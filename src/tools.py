"""The agent's TOOLS: small, typed functions over free, keyless crypto data.

Each function does exactly one job, returns a plain Python dict (JSON-serialisable),
and is paired with a JSON Schema in ``TOOL_SCHEMAS`` so the model knows when and how
to call it. Keeping the schema next to the implementation is deliberate: the two must
agree, and a reader should be able to see both at once.

Two free, public, no-key data sources back these tools:

- **Binance** public REST (``api.binance.com``) for spot prices and daily klines.
- **DeFiLlama** public REST (``api.llama.fi``) for protocol total-value-locked (TVL).

Neither needs an API key. We add short timeouts and turn HTTP errors into structured
``{"error": ...}`` payloads, because a tool that raises will crash the agent loop,
whereas a tool that returns an error lets the model read it and adapt.
"""
from __future__ import annotations

import os

import requests

_BINANCE = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
_LLAMA = "https://api.llama.fi"
_TIMEOUT = 15


def _norm_symbol(symbol: str) -> str:
    """Accept friendly inputs ('btc', 'BTC', 'BTCUSDT') and return a Binance pair."""
    s = symbol.strip().upper()
    if s.endswith(("USDT", "USDC", "BUSD", "FDUSD")):
        return s
    return s + "USDT"


def get_price(symbol: str) -> dict:
    """Latest spot price and 24h move for one trading pair, from Binance.

    Returns a dict with ``symbol``, ``price`` (float), ``pct_change_24h`` (float),
    ``high_24h``, ``low_24h`` and the quote currency. On failure returns ``{"error": ...}``.
    """
    pair = _norm_symbol(symbol)
    try:
        r = requests.get(f"{_BINANCE}/api/v3/ticker/24hr",
                         params={"symbol": pair}, timeout=_TIMEOUT)
        r.raise_for_status()
        d = r.json()
    except requests.RequestException as e:
        return {"error": f"price lookup failed for {pair}: {e}"}
    except ValueError as e:  # bad JSON
        return {"error": f"could not parse price response for {pair}: {e}"}
    if "lastPrice" not in d:
        return {"error": f"no price data for {pair} (is the symbol valid?)"}
    quote = pair[-4:] if pair.endswith(("USDT", "USDC", "BUSD")) else pair[-4:]
    return {
        "symbol": pair,
        "price": float(d["lastPrice"]),
        "pct_change_24h": float(d["priceChangePercent"]),
        "high_24h": float(d["highPrice"]),
        "low_24h": float(d["lowPrice"]),
        "quote_currency": quote,
    }


def get_klines_summary(symbol: str, days: int = 30) -> dict:
    """Summarise the last ``days`` daily candles for a pair, from Binance.

    Rather than dumping raw OHLCV (which would flood the model's context), this
    returns a compact summary: start/end close, total return, annualised volatility
    of daily returns, and the highest high / lowest low over the window. Summarising
    inside the tool is the honest way to keep an agent's context small and cheap.
    """
    pair = _norm_symbol(symbol)
    days = max(2, min(int(days), 365))  # Binance caps at 1000; keep it sane for a summary
    try:
        r = requests.get(f"{_BINANCE}/api/v3/klines",
                         params={"symbol": pair, "interval": "1d", "limit": days},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        rows = r.json()
    except requests.RequestException as e:
        return {"error": f"klines lookup failed for {pair}: {e}"}
    except ValueError as e:
        return {"error": f"could not parse klines for {pair}: {e}"}
    if not rows:
        return {"error": f"no klines for {pair} (is the symbol valid?)"}

    closes = [float(row[4]) for row in rows]
    highs = [float(row[2]) for row in rows]
    lows = [float(row[3]) for row in rows]
    # Daily simple returns and their standard deviation, annualised by sqrt(365).
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    mean = sum(rets) / len(rets) if rets else 0.0
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1) if len(rets) > 1 else 0.0
    vol_annualised = (var ** 0.5) * (365 ** 0.5)
    return {
        "symbol": pair,
        "days": len(closes),
        "start_close": round(closes[0], 6),
        "end_close": round(closes[-1], 6),
        "return_pct": round((closes[-1] / closes[0] - 1.0) * 100, 2),
        "annualised_vol_pct": round(vol_annualised * 100, 2),
        "high": round(max(highs), 6),
        "low": round(min(lows), 6),
    }


def get_tvl(protocol: str) -> dict:
    """Current total value locked (TVL) for a DeFi protocol, from DeFiLlama.

    ``protocol`` is a DeFiLlama slug (e.g. 'aave', 'lido', 'uniswap', 'makerdao').
    Returns ``{"protocol": ..., "tvl_usd": float}`` or ``{"error": ...}``.
    """
    slug = protocol.strip().lower().replace(" ", "-")
    try:
        r = requests.get(f"{_LLAMA}/tvl/{slug}", timeout=_TIMEOUT)
        r.raise_for_status()
        value = r.json()
    except requests.RequestException as e:
        return {"error": f"TVL lookup failed for '{slug}': {e}"}
    except ValueError as e:
        return {"error": f"could not parse TVL for '{slug}': {e}"}
    # DeFiLlama returns a bare number for a known slug, or an error object/string otherwise.
    if not isinstance(value, (int, float)):
        return {"error": f"unknown protocol slug '{slug}' (try aave, lido, uniswap, ...)"}
    return {"protocol": slug, "tvl_usd": float(value)}


# --- JSON Schemas: the contract the model sees -------------------------------
# These mirror the functions above. The descriptions are prescriptive about WHEN to
# call each tool, which measurably improves how reliably the model reaches for them.
TOOL_SCHEMAS = [
    {
        "name": "get_price",
        "description": (
            "Get the latest spot price and 24-hour move for one crypto trading pair "
            "from Binance. Call this whenever the user asks about the current or live "
            "price of a coin, or how it has moved today. Accepts a coin symbol like "
            "'BTC', 'ETH' or a full pair like 'BTCUSDT'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Coin symbol or Binance pair, e.g. 'BTC', 'ETH', 'SOLUSDT'.",
                }
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_klines_summary",
        "description": (
            "Summarise recent daily price action for a crypto pair from Binance: total "
            "return, annualised volatility, and the high/low over a window of days. Call "
            "this when the user asks about recent performance, trend, returns or how "
            "volatile a coin has been over a period (not just the live price)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Coin symbol or Binance pair, e.g. 'BTC', 'ETH', 'SOLUSDT'.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of daily candles to summarise (2-365). Default 30.",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_tvl",
        "description": (
            "Get the current total value locked (TVL), in US dollars, for a DeFi "
            "protocol from DeFiLlama. Call this when the user asks how much value is "
            "locked in, or the size of, a DeFi protocol such as Aave, Lido, Uniswap or "
            "MakerDAO. Takes a DeFiLlama protocol slug (lowercase, e.g. 'aave')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protocol": {
                    "type": "string",
                    "description": "DeFiLlama protocol slug, e.g. 'aave', 'lido', 'uniswap'.",
                }
            },
            "required": ["protocol"],
        },
    },
]

# A name -> callable registry, so the agent loop can dispatch a tool_use block by name.
TOOL_FUNCTIONS = {
    "get_price": get_price,
    "get_klines_summary": get_klines_summary,
    "get_tvl": get_tvl,
}


def run_tool(name: str, tool_input: dict) -> dict:
    """Dispatch a tool call by name. Unknown tools return a structured error."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    return fn(**tool_input)
