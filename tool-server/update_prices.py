#!/usr/bin/env python3
"""Fetch current model pricing and write to usage_prices.json.

Runs monthly via cron. Output goes to /app/backend/data/usage_prices.json
on the tool-server data volume (bind-mounted from host).

Sources:
  Fireworks: https://fireworks.ai/pricing (HTML scrape)
  DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing (HTML scrape)
  OpenAI:    hardcoded (stable, announced months in advance)
  Anthropic: hardcoded (stable, announced months in advance)

All prices in USD per million tokens (input, output) for serverless inference.
"""

import json
import logging
import os
import re
import sys
from urllib.request import urlopen, Request

logger = logging.getLogger(__name__)

OUTPUT = os.getenv("PRICE_OUTPUT", "/app/backend/data/usage_prices.json")

# ── Fireworks ──────────────────────────────────────────────────────────────
# DeepSeek models on Fireworks (serverless)
FIREWORKS_DEEPSEEK = {
    "deepseek-v4-pro": (1.74, 3.48),
    "deepseek-v4-flash": (0.22, 0.88),
    "deepseek-v3p1": (0.27, 1.10),
    "deepseek-v3p2": (0.27, 1.10),
}

# Other Fireworks models
FIREWORKS_OTHER = {
    "glm-5p1": (1.40, 4.40),
    "glm-5": (1.40, 4.40),
    "kimi-k2p6": (0.95, 4.00),
    "kimi-k2p5": (0.95, 4.00),
    "kimi-k2-thinking": (0.95, 4.00),
    "gpt-oss-120b": (0.15, 0.60),
    "gpt-oss-20b": (0.05, 0.20),
    "mixtral-8x22b-instruct": (0.90, 0.90),
    "cogito-671b-v2-p1": (0.90, 0.90),
    "qwenvl2.5": (0.02, 0.0),  # embedding
}

# ── DeepSeek-direct ────────────────────────────────────────────────────────
# These are typically cheaper than Fireworks-proxied.
DEEPSEEK_DIRECT = {
    "deepseek-v4-pro": (1.10, 2.20),
    "deepseek-v4-flash": (0.14, 0.56),
    "deepseek-v3p1": (0.18, 0.72),
}

# ── Premium prose tiers ────────────────────────────────────────────────────
PREMIUM = {
    "gpt-5.5": (1.25, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

# ── Aliases ────────────────────────────────────────────────────────────────
ALIASES = {
    "v4-research-writing": "deepseek-v4-pro",
    "qwen3-embedding": "qwenvl2.5",
}


def _scrape_fireworks() -> dict | None:
    """Try to fetch live Fireworks pricing. Falls back to hardcoded."""
    try:
        req = Request("https://fireworks.ai/pricing",
                      headers={"User-Agent": "PrismAI-price-fetcher/1.0"})
        html = urlopen(req, timeout=15).read().decode()
    except Exception as e:
        logger.warning("Fireworks pricing fetch failed: %s", e)
        return None

    prices = {}
    # Match model name + price pattern: e.g., "DeepSeek V4 Pro  $1.74 / $3.48"
    for pattern, key in [
        (r"DeepSeek\s+V4\s+Pro.*?\$(\d+\.?\d*)\s*/\s*\$(\d+\.?\d*)", "deepseek-v4-pro"),
        (r"DeepSeek\s+V4\s+Flash.*?\$(\d+\.?\d*)\s*/\s*\$(\d+\.?\d*)", "deepseek-v4-flash"),
        (r"GLM-5\.1.*?\$(\d+\.?\d*)\s*/\s*\$(\d+\.?\d*)", "glm-5p1"),
        (r"Kimi\s+K2\.[56].*?\$(\d+\.?\d*)\s*/\s*\$(\d+\.?\d*)", "kimi-k2p6"),
    ]:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            prices[key] = (float(m.group(1)), float(m.group(2)))
    return prices if prices else None


def build_prices():
    """Merge hardcoded defaults with live-scraped Fireworks prices."""
    prices = {}

    # Fireworks
    fw = _scrape_fireworks()
    prices.update(FIREWORKS_OTHER)
    if fw:
        prices.update(fw)
        logger.info("Live Fireworks prices applied for: %s", list(fw.keys()))
    else:
        prices.update(FIREWORKS_DEEPSEEK)
        logger.info("Falling back to hardcoded Fireworks prices")

    # Premium & aliases
    prices.update(PREMIUM)
    for alias, target in ALIASES.items():
        if target in prices and alias not in prices:
            prices[alias] = prices[target]

    return prices


def main():
    prices = build_prices()
    # Convert tuples to lists for JSON
    json_prices = {k: list(v) for k, v in prices.items()}
    with open(OUTPUT, "w") as f:
        json.dump(json_prices, f, indent=2)
    logger.info("Wrote %d model prices to %s", len(json_prices), OUTPUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
