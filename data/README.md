# Data fixtures

The only file here is `cached_agent_run.json`: a clearly labelled synthetic, illustrative
Claude transcript, used as the cached fallback when no `ANTHROPIC_API_KEY` is set so the
notebook reads end to end without a key.

The data tools themselves (Binance, DeFiLlama) are free, keyless and run live -- including in
CI, where `BINANCE_BASE_URL` points at the public `data-api.binance.vision` mirror. Only the
Claude reasoning step is cached. Nothing here is real API output; the figures in the cached
transcript are plausible but illustrative. Public market data only.
