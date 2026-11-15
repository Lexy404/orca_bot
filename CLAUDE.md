# Orca Bot

Solana CLMM trading system for the Orca Whirlpools SOL/USDC 0.04% pool (tick spacing 4). A Pre-LN Transformer predicts the hourly price range [p_low, p_high] and trend; a downside-aware policy decides whether to provide liquidity, skew the range down, or sit in cash. Inference is served over ZeroMQ to a separate on-chain executor.

Stack: Python 3.10+, PyTorch, NumPy, Pandas, pyzmq.

The project is under active development. The code is the source of truth: read it for constants, thresholds, feature lists and loss details instead of relying on this file. If you change behavior described here, update this file in the same change.

## Scope
- Work only on the Python ML/inference code in this session.
- The TypeScript folder `orca_executor` is out of scope. Do not read, edit or refactor it unless explicitly asked.

## Commands
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch numpy pandas pyzmq requests (+ plotly/ipykernel for the notebook)

python ml/fetch_data.py          # download Bybit SOLUSDC 1h candles (backs up existing CSV)
python ml/pretrain.py            # self-supervised MAE pre-training
python ml/train.py               # supervised fine-tuning (with embargo); best checkpoint -> weights/runs/<ts>/
python ml/train.py --promote     # same, then back up and replace weights/orca_transformer.pth
python ml/backtest_historical.py [--weights PATH] [--pool-liquidity L] [--volume-multiplier M]  # out-of-sample backtest -> backtest_results.csv
python ml/live_predict.py        # ZeroMQ inference server
python ml/dataset.py             # feature / dataset integrity check
python tests/test_feature_parity.py # feature parity, splits, quantile ordering, live handler errors, CLMM math, loss
```

After changing `model.py` or `dataset.py`, run a forward pass on a random batch and check that output shape is `[batch, 3]` and `pred_high > pred_low` for every row.

## Files
Layout: `ml/` model, training, backtest and live server code (flat imports inside the folder, run scripts from the project root); `tests/` tests only (`test_feature_parity.py`, manual ZeroMQ client `test_client.py`); the on-chain data tools `collect_pool_snapshots.py` and `analyze_pool_volume.py` stay in the root because the LaunchAgent points at them. Project root = parent of `ml/`; `data/`, `weights/`, `logs/` and outputs live in the root.
- `ml/config.py`: shared constants (`NUM_FEATURES`, `SEQ_LEN`, model output scales, regime thresholds, loss weights, `skewed_bounds`, `validate_bounds`, live limits, `SEED`). Change them only here.
- `ml/model.py`: `OrcaCryptoTransformer`, `OrcaMaskedAutoencoder`
- `ml/dataset.py`: feature engineering, sliding-window dataset, `build_single_window_features` (live path)
- `ml/fetch_data.py`: downloads hourly SOLUSDC candles (default Bybit spot, same market as the live executor; volume in SOL; only closed candles)
- `ml/pretrain.py`, `ml/train.py`: MAE pre-training, quantile fine-tuning
- `ml/backtest_historical.py`: CLMM simulator with fees and friction
- `ml/live_predict.py`: ZeroMQ REP server and regime dispatch
- `collect_pool_snapshots.py`: read-only hourly snapshot of the on-chain pool (liquidity profile, fee growth) -> `data/pool_snapshots.csv`; scheduled hourly by a macOS LaunchAgent (template: `launchd/orca_pool_snapshots.plist.example`). Goal: replace the backtest volume multiplier with measured data
- `analyze_pool_volume.py`: read-only; measured pool volume from `data/pool_snapshots.csv` vs Bybit SOLUSDC (public candles fetched on the fly, `data/sol_usdc_data.csv` untouched); prints per-interval table and the pool/Bybit ratio. Run it at each checkpoint
- `weights/*.pth`, `weights/runs/`, `data/*.csv`: artifacts, see Safety. `performance_history.csv` (executor output, read by the notebook) and `backtest_results.csv` (backtest output) are git-ignored.

## Invariants (must hold, whatever the current implementation is)

**No temporal leakage**
- Splits are `split_ranges` in `config.py`: train | embargo | val | embargo | test. Test is used only by the backtest; invalid windows (time gaps) are skipped via `valid_indices`.
- Validation data must never overlap the training window or the model's input history. The embargo between train and validation must cover at least input window length + 1 forecast step.
- Never shuffle time series outside strictly separated splits.
- Every rolling/EMA indicator uses past data only.

**Training and serving use identical features**
- `NUM_FEATURES` lives in `config.py` and is imported everywhere; never hardcode 11 or 72.
- The offline feature pipeline and `build_single_window_features` must produce the same features in the same order. Change both together, plus the input embedding.

**Non-crossing quantiles**
- `pred_high` must exceed `pred_low` by at least `min_spread`, guaranteed by the model's parameterization (Softplus), not by post-hoc clipping. Do not remove it.

**CLMM math**
- Guard against `pa >= pb` and `p0 <= 0` (see the guards used in `backtest_historical.py`).
- One-sided allocation: price at or below `pa` means 100% SOL, at or above `pb` means 100% USDC.

**ZeroMQ**
- `recv_string()` and `send_string()` stay in strict lockstep (REQ/REP).
- All parse and inference errors still return a valid JSON error response so the client never hangs.

**Devices**
```python
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
```

## Known issues
These are bugs, not intended behavior. Do not copy or reinforce them; fix when asked, and tell me if you hit one.
<!-- Keep to 3-5 items, remove when fixed. Example format:
- `dataset.py`: possible lookahead in `chaikin_vol`; verify before relying on it.
-->
- Backtest fee income uses an on-chain snapshot of the pool's active liquidity (2026-11-14, `POOL_ACTIVE_LIQUIDITY`; it varies ~4.3-5.8M between ticks) and Bybit volume scaled by `--volume-multiplier`. Orca API (2026-11-14): pool volume was ~3.5x Bybit SOLUSDC over 7d and ~5.6x over 30d (no hourly history available). Break-even is near x3.5 and profit grows steeply above it, so absolute profit is only indicative; compare variants at the same settings. The pool/Bybit ratio is NOT stable (~3.9x over 7d but ~25-30x on 2026-11-14/15, measured with `analyze_pool_volume.py`), so a fixed multiplier is a poor model; replace it with measured volume/liquidity.
- `train.py` loss is asymmetric (`config.LOSS_*`): the IL penalty puts the effective low quantile near 5% and `q_trend=0.42` biases the trend down, so more hours go to cash/skew and revenue dropped in the backtest. Not retuned yet.
- Backtest does not quantize bounds to tick spacing (4, i.e. 0.04% steps: negligible next to `MIN_SPREAD`).

## Safety
- Never read, print, log or commit private keys, wallet files, `.env` or RPC credentials.
- Never send on-chain transactions or touch mainnet. Default to backtest/simulation; real execution needs explicit human confirmation.
- Do not overwrite or delete `weights/*.pth` or `data/*.csv` without being asked (`train.py` only replaces production weights with `--promote`, after a backup). Do not print whole datasets.
- Do not start full pre-training or training runs unless asked; they are long-running.

## Regimes
`live_predict.py` maps `pred_trend_coef` to one of three regimes. Thresholds and the skew formula live in `config.py` (shared by train monitoring, backtest and live).
- `SKIPPED_CASH`: sharp predicted drop; withdraw LP, hold 100% USDC, `skip_hour: true`.
- `SKEW_RANGE_DOWN`: moderate predicted drop; defensive LP with the range shifted down.
- `NORMAL_LP`: otherwise; concentrate liquidity on the predicted bounds.

Backtest and live import the same constants; keep it that way.

## ZeroMQ protocol
Server binds `tcp://127.0.0.1:5555` (override with env `ORCA_ZMQ_HOST`); clients connect to `tcp://localhost:5555`. Requires `MIN_LIVE_CANDLES` (= 72 + 78 warm-up, see `config.py`) strictly consecutive 1h candles with `time` in seconds, so EMA/ROC indicators match training; `current_price` must be within 5% of the last close, candles must pass OHLCV sanity checks, and the last candle must be at most 3h old (`ORCA_ALLOW_STALE=1` disables this for testing only). Schemas, as of now:

Request:
```json
{"current_price": 182.45,
 "candles": [{"time": 1731315600, "open": 181.20, "high": 183.10,
              "low": 180.95, "close": 182.45, "volume": 24500.12}]}
```
Response:
```json
{"status": "success", "regime": "NORMAL_LP", "skip_hour": false,
 "current_price": 182.45, "pred_low": 181.1250, "pred_high": 184.3410,
 "pred_trend_coef": 1.002154, "raw_spread_pct": 1.763, "price_in_range": true}
```
