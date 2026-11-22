"""
Front Office - Servicio de Noticias Financieras y Sentimiento (Google News RSS + Lexicón).

Proporciona señal ortogonal al precio (catalizadores FOMC/Earnings/SEC y sesgo de sentimiento)
con caché TTL de 10 minutos, timeout estricto y modo neutral instantáneo en Historical Replay.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import httpx

BULLISH_WORDS = {
    "surge", "rally", "beat", "beats", "upgrade", "bullish", "growth", "record",
    "profit", "approval", "etf", "outperform", "buy", "strong", "gain", "gains",
    "soar", "soars", "rebound", "breakout", "partnership", "expansion", "raised",
}
BEARISH_WORDS = {
    "plunge", "drop", "crash", "miss", "misses", "downgrade", "bearish", "lawsuit",
    "fraud", "hack", "sec", "ban", "fine", "investigation", "cut", "cuts", "layoff",
    "warning", "loss", "losses", "decline", "slump", "recession", "risk", "selloff",
}
CATALYST_KEYWORDS = {
    "earnings", "fomc", "fed", "rate", "inflation", "cpi", "sec", "lawsuit",
    "downgrade", "upgrade", "guidance", "hack", "etf", "dividend", "merger", "antitrust",
}

COMPANY_QUERY_HINTS = {
    "BTC-USD": "Bitcoin crypto BTC",
    "BTCUSDT": "Bitcoin crypto BTC",
    "SPY": "S&P 500 SPY stock market",
    "QQQ": "Nasdaq QQQ tech stocks",
}


class NewsSentimentService:
    """Recolector ligero de titulares financieros + analizador de sentimiento y catalizadores."""

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

    @staticmethod
    def score_headlines(headlines: List[str]) -> Dict[str, Any]:
        if not headlines:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "catalysts": [],
                "headline_count": 0,
            }

        scores: List[float] = []
        catalysts_found: set[str] = set()
        for title in headlines:
            tokens = {w.strip(".,:;!?()[]{}\"'").lower() for w in title.split()}
            pos = len(tokens & BULLISH_WORDS)
            neg = len(tokens & BEARISH_WORDS)
            for cat in tokens & CATALYST_KEYWORDS:
                catalysts_found.add(cat.upper())
            if pos + neg > 0:
                scores.append((pos - neg) / float(pos + neg))
            else:
                scores.append(0.0)

        avg_score = sum(scores) / float(len(scores)) if scores else 0.0
        if avg_score >= 0.15:
            label = "BULLISH"
        elif avg_score <= -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            "sentiment_score": round(avg_score, 3),
            "sentiment_label": label,
            "catalysts": sorted(catalysts_found)[:6],
            "headline_count": len(headlines),
        }

    def get_symbol_context(
        self,
        symbol: str,
        source_mode: str = "replay",
        fetch_live: bool = False,
    ) -> Dict[str, Any]:
        if source_mode == "replay" and not fetch_live:
            return {
                "symbol": symbol,
                "source": "replay_neutral",
                "sentiment_score": 0.0,
                "sentiment_label": "NEUTRAL",
                "catalysts": [],
                "headlines": [],
                "headline_count": 0,
                "high_impact_event": False,
            }

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(symbol)
            if cached is not None and (now - cached[0]) < self.ttl_seconds:
                return dict(cached[1])

        query = COMPANY_QUERY_HINTS.get(symbol, f"{symbol} stock")
        url = (
            "https://news.google.com/rss/search?"
            f"q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        )
        headlines: List[str] = []
        source_tag = "google_news_rss"
        try:
            with httpx.Client(timeout=2.5, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:5]:
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    headlines.append(title_el.text.strip())
        except Exception:
            source_tag = "rss_unavailable_neutral"

        scored = self.score_headlines(headlines)
        payload = {
            "symbol": symbol,
            "source": source_tag,
            "sentiment_score": scored["sentiment_score"],
            "sentiment_label": scored["sentiment_label"],
            "catalysts": scored["catalysts"],
            "headlines": headlines[:4],
            "headline_count": len(headlines),
            "high_impact_event": len(scored["catalysts"]) > 0,
        }
        with self._lock:
            self._cache[symbol] = (time.monotonic(), payload)
        return payload
