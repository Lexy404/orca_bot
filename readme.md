# Orca CLMM Range Predictor

Research code for managing concentrated liquidity on the **Orca Whirlpools SOL/USDC 0.04%** pool (Solana, tick spacing 4). A Pre-LN Transformer predicts the next hour's price range `[p_low, p_high]` and a trend coefficient from hourly candles; a downside-aware policy then decides whether to provide liquidity on the predicted range, skew it down, or sit in cash. Inference is served over ZeroMQ.

> **Status: research project, not a profitable trading system.** The backtest shows that the strategy only pays off when pool volume relative to active liquidity is high enough (see [Findings](#findings)). Nothing here sends on-chain transactions; the repository contains no keys and no executor.

## How it works

1. **Self-supervised pre-training** (`pretrain.py`, `model.py`): a masked autoencoder with contiguous block masking on hourly SOLUSDC features (OHLCV, log return, high-low range, realized volatility, NATR, Chaikin volatility, CLV, plus hour/day embeddings).
2. **Quantile fine-tuning** (`train.py`): pinball losses for the range bounds, an impermanent-loss downside penalty, a range-width regularizer and an asymmetric trend quantile loss. All weights live in `config.py`.
   - Non-crossing bounds are guaranteed by the parameterization, not by clipping: `p_high = p_low + min_spread + softplus(...)`.
3. **Regime policy** (`live_predict.py`, thresholds in `config.py`): `NORMAL_LP`, `SKEW_RANGE_DOWN` (moderate predicted drop, range shifted down) or `SKIPPED_CASH` (sharp predicted drop, hold USDC).
4. **Backtest** (`backtest_historical.py`): CLMM simulator with fees, impermanent loss and friction, run on a held-out test split.
5. **Pool data collection** (`collect_pool_snapshots.py`, `analyze_pool_volume.py`): hourly read-only snapshots of the on-chain pool (active liquidity, fee growth). Pool volume is derived from the fee-growth delta, so the backtest's volume assumption can be replaced by measured data.

### Integrity rules the code is built around

- **No temporal leakage**: train | embargo | val | embargo | test splits (`config.split_ranges`), embargo at least window length + 1 step, indicators use past data only.
- **Train/serve parity**: the offline pipeline and `build_single_window_features` (live path) must produce identical features; `test_feature_parity.py` checks this.
- **CLMM math guards**: `pa >= pb` and `p0 <= 0` are rejected; one-sided allocation below/above the range.
- **ZeroMQ lockstep**: every request, including malformed ones, gets a valid JSON reply.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py            # ~2 years of hourly SOLUSDC candles (Bybit spot)
python pretrain.py              # MAE pre-training
python train.py                 # fine-tuning, checkpoint -> weights/runs/<timestamp>/
python backtest_historical.py   # out-of-sample backtest -> backtest_results.csv
python test_feature_parity.py   # integrity tests
python live_predict.py          # ZeroMQ inference server on tcp://127.0.0.1:5555
```

`data/`, `weights/` and `logs/` are git-ignored; generate them with the commands above. `backtest_analytics.ipynb` visualizes the backtest output.

### Pool snapshots (optional)

```bash
python collect_pool_snapshots.py      # one read-only snapshot -> data/pool_snapshots.csv
python analyze_pool_volume.py         # measured pool volume vs Bybit volume
```

To collect hourly on macOS, copy `launchd/orca_pool_snapshots.plist.example`, replace the placeholder paths, and load it with `launchctl`. Public RPC endpoints are used by default (override with `ORCA_SNAPSHOT_RPC`).

## ZeroMQ protocol

The server binds `tcp://127.0.0.1:5555` (override with `ORCA_ZMQ_HOST`). It needs `MIN_LIVE_CANDLES` strictly consecutive hourly candles (`time` in seconds) so that indicators match training.

```json
{"current_price": 182.45,
 "candles": [{"time": 1731315600, "open": 181.20, "high": 183.10,
              "low": 180.95, "close": 182.45, "volume": 24500.12}]}
```

```json
{"status": "success", "regime": "NORMAL_LP", "skip_hour": false,
 "current_price": 182.45, "pred_low": 181.1250, "pred_high": 184.3410,
 "pred_trend_coef": 1.002154, "raw_spread_pct": 1.763, "price_in_range": true}
```

## Findings

- Per active hour (Bybit volume x1): fees about +0.025%, impermanent loss about -0.074%, costs about -0.012%. Break-even needs pool volume of roughly 3.5x Bybit SOLUSDC volume, and most losses come from hours where price leaves the range downward.
- Measured on-chain volume matches Orca's own API (about $133M/day extrapolated from 10 hours vs $132M reported for 24 h), so the fee-growth derivation looks correct.
- The pool/Bybit volume ratio is **not stable**: about 3.9x over 7 days, 5.6x over 30 days, but roughly 25-30x in the most recent day. A fixed volume multiplier is therefore a weak model, and the backtest's absolute profit is only indicative.
- Loss-parameter variants barely change backtest results; a symmetric trend quantile collapses the trend head so cash/skew regimes never trigger.

## Roadmap

- Replace the volume multiplier by measured volume / active liquidity (needs weeks of snapshots).
- Test wider ranges and re-ranging only when price leaves the range.
- Add an "LP only when expected fees exceed expected IL plus costs" filter based on trailing pool volume, validated on a held-out period.

## Limitations

- Backtest fee income depends on assumed pool volume and a liquidity snapshot; absolute numbers are indicative, compare variants at identical settings.
- The backtest does not quantize bounds to tick spacing (0.04% steps).
- No transaction execution is included. The on-chain executor used during development is a separate private component.

## License

MIT, see [LICENSE](LICENSE).
