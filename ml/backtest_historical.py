import argparse
import os
import torch
import numpy as np
import pandas as pd
from dataset import CryptoTimeSeriesDataset
from model import OrcaCryptoTransformer
from config import NUM_FEATURES, SEQ_LEN, CRITICAL_DROP_THRESHOLD, MODERATE_DROP_THRESHOLD, skewed_bounds, split_ranges, validate_bounds

# Backtest Configuration
BASE_CAPITAL_USD = 1000.0

# Trading Friction & CLMM Constants (Orca Whirlpool SOL/USDC 0.04% pool)
FEE_RATE = 0.0004          # 0.04% pool fee (on-chain feeRate of Czfq3xZZ..., tick spacing 4)
PROTOCOL_FEE_SHARE = 0.13  # protocolFeeRate: 13% of the fee goes to the protocol, LPs get 87%
SWAP_FEE_PCT = 0.0005      # 0.05% swap fee, charged only on the notional actually swapped when (re)entering
GAS_FEE_USD = 0.005        # ~$0.005 Solana transaction cost per transaction
POOL_ACTIVE_LIQUIDITY = 4_319_490.0  # on-chain active liquidity L (human sqrt units) of the pool at the current tick,
                                     # read once on 2026-11-14 (TVL ~$16.2M); historical liquidity differed
VOLUME_MULTIPLIER = 1.0              # Orca pool volume / Bybit SOLUSDC volume; measured 2026-11: ~3.5 (7d) .. ~5.6 (30d). 1.0 = Bybit as is



def compute_clm_position(p0, pa, pb, capital):
    """
    Computes liquidity L and token balances (x0: SOL, y0: USDC) for concentrated liquidity.
    """
    if p0 <= 0 or pa <= 0 or pa >= pb:
        raise ValueError(f"Invalid CLMM bounds: p0={p0}, pa={pa}, pb={pb}")
    sqrt_p0 = np.sqrt(p0)
    sqrt_pa = np.sqrt(pa)
    sqrt_pb = np.sqrt(pb)

    if p0 <= pa:
        # 100% SOL position
        l = capital / (p0 * (1.0 / sqrt_pa - 1.0 / sqrt_pb))
        x0 = l * (1.0 / sqrt_pa - 1.0 / sqrt_pb)
        y0 = 0.0
    elif p0 >= pb:
        # 100% USDC position
        l = capital / (sqrt_pb - sqrt_pa)
        x0 = 0.0
        y0 = capital
    else:
        # Two-sided concentrated liquidity
        l = capital / ((sqrt_p0 - sqrt_pa) + p0 * (1.0 / sqrt_p0 - 1.0 / sqrt_pb))
        x0 = l * (1.0 / sqrt_p0 - 1.0 / sqrt_pb)
        y0 = l * (sqrt_p0 - sqrt_pa)

    return l, x0, y0

def evaluate_clm_hour(p1, pa, pb, l, x0, y0, volume_usd, candle_high, candle_low, pool_liquidity, volume_multiplier=1.0):
    """
    Evaluates end-of-hour position value, impermanent loss and volume-based fee income.

    Fee model: fees = volume * multiplier * FEE_RATE * (1 - PROTOCOL_FEE_SHARE) * L / (L + pool_liquidity)
    * (share of the candle's high-low span that lies inside [pa, pb]). pool_liquidity is the pool's active
    liquidity at the current tick (a fixed on-chain snapshot), volume is Bybit's as a proxy for Orca's.
    Returns (x1, y1, v1, hodl_value, il_usd, fee_usd).
    """
    sqrt_p1 = np.sqrt(p1)
    sqrt_pa = np.sqrt(pa)
    sqrt_pb = np.sqrt(pb)

    if p1 <= pa:
        x1 = l * (1.0 / sqrt_pa - 1.0 / sqrt_pb)
        y1 = 0.0
    elif p1 >= pb:
        x1 = 0.0
        y1 = l * (sqrt_pb - sqrt_pa)
    else:
        x1 = l * (1.0 / sqrt_p1 - 1.0 / sqrt_pb)
        y1 = l * (sqrt_p1 - sqrt_pa)

    v1 = x1 * p1 + y1
    hodl_value = x0 * p1 + y0
    il_usd = v1 - hodl_value

    span = candle_high - candle_low
    if span > 0:
        overlap = max(0.0, min(candle_high, pb) - max(candle_low, pa)) / span
    else:
        overlap = 1.0 if pa <= candle_high <= pb else 0.0
    share = l / (l + pool_liquidity)
    fee_usd = volume_usd * volume_multiplier * FEE_RATE * (1.0 - PROTOCOL_FEE_SHARE) * share * overlap

    return x1, y1, v1, hodl_value, il_usd, fee_usd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="checkpoint to evaluate (default: weights/orca_transformer.pth)")
    ap.add_argument("--pool-liquidity", type=float, default=POOL_ACTIVE_LIQUIDITY,
                    help="active pool liquidity L (human sqrt units) used as the fee-share denominator")
    ap.add_argument("--volume-multiplier", type=float, default=VOLUME_MULTIPLIER,
                    help="Orca pool volume as a multiple of Bybit SOLUSDC volume")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Running offline historical CLMM backtest on device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "sol_usdc_data.csv")
    weights_path = args.weights or os.path.join(base_dir, "weights", "orca_transformer.pth")

    raw_df = pd.read_csv(csv_path)
    dataset = CryptoTimeSeriesDataset(csv_file=csv_path, history_len=SEQ_LEN)
    total_samples = len(dataset)

    # Held-out test split: after train and val (with embargo), never used for training or checkpoint selection
    test_indices = dataset.valid_indices(split_ranges(total_samples)["test"])
    start_idx = test_indices[0]
    TEST_HOURS = len(test_indices)
    print(f"Loaded dataset ({total_samples} windows). Evaluating held-out test split: {TEST_HOURS} windows (indices [{start_idx} .. {total_samples - 1}]).")
    print(f"Pool active liquidity L={args.pool_liquidity:,.0f} | volume multiplier x{args.volume_multiplier} (fee income scales with the multiplier)")

    model = OrcaCryptoTransformer(num_features=NUM_FEATURES, seq_len=SEQ_LEN).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded trained Transformer weights from: {weights_path}\n")

    records = []
    capital = BASE_CAPITAL_USD
    held_sol = 0.0       # SOL held at the start of the hour (after the previous hour closed)
    prev_active = False  # was an LP position open during the previous hour

    with torch.no_grad():
        for i in test_indices:
            batch = dataset[i]
            x_tensor = batch[0].unsqueeze(0).to(device)
            hours = batch[2].unsqueeze(0).to(device)
            days = batch[3].unsqueeze(0).to(device)

            preds = model(x_tensor, hours, days)[0].cpu().numpy()
            pred_high_coef, pred_low_coef, pred_trend_coef = preds[0], preds[1], preds[2]

            # Candle data
            candle_idx = i + SEQ_LEN - 1
            current_price = raw_df.loc[candle_idx, "close"]
            next_candle = raw_df.loc[candle_idx + 1]
            next_high = next_candle["high"]
            next_low = next_candle["low"]
            next_close = next_candle["close"]
            volume_usd = next_candle["volume"] * next_close

            # Unscaled tick prices
            pred_high = current_price * pred_high_coef
            pred_low = current_price * pred_low_coef

            # Dynamic Regime Dispatcher
            go_cash = pred_trend_coef < CRITICAL_DROP_THRESHOLD
            if not go_cash:
                if pred_trend_coef < MODERATE_DROP_THRESHOLD:
                    regime = "ACTIVE_LP (SKEWED_DOWN)"
                    pa, pb = skewed_bounds(current_price, pred_low, pred_high)
                else:
                    regime = "ACTIVE_LP (NORMAL)"
                    pa, pb = pred_low, pred_high
                try:
                    validate_bounds(pa, pb, current_price)
                except ValueError:
                    go_cash = True  # unusable range -> stay in cash instead of aborting the backtest

            start_capital = capital
            fee_usd = 0.0
            il_usd = 0.0
            in_range = False

            if go_cash:
                if pred_trend_coef < CRITICAL_DROP_THRESHOLD:
                    regime = "SKIPPED_CASH (100% USDC)"
                else:
                    regime = "SKIPPED_CASH (INVALID_RANGE)"
                pa, pb = current_price, current_price
                status = "CASH_PRESERVATION"
                # Exit costs: swap the SOL held (if any) back to USDC, one tx if a position was open
                swap_notional = held_sol * current_price
                capital -= SWAP_FEE_PCT * swap_notional + (GAS_FEE_USD if prev_active else 0.0)
                held_sol = 0.0
                prev_active = False
                hourly_ret_pct = 0.0
            else:
                in_range = (next_low >= pa) and (next_high <= pb)
                if in_range:
                    status = "IN_RANGE"
                elif next_high > pb:
                    status = "OUT_OF_RANGE_HIGH"
                else:
                    status = "OUT_OF_RANGE_LOW"

                # Entry costs: swap only the imbalance between current holdings and the new position's SOL need
                cap_for_pos = capital
                _, x_need, _ = compute_clm_position(current_price, pa, pb, cap_for_pos)
                swap_notional = abs(held_sol - x_need) * current_price
                txs = 2 if prev_active else 1  # close old + open new
                capital -= SWAP_FEE_PCT * swap_notional + GAS_FEE_USD * txs

                l, x0, y0 = compute_clm_position(current_price, pa, pb, capital)
                x1, y1, v1, hodl_val, il_usd, fee_usd = evaluate_clm_hour(
                    next_close, pa, pb, l, x0, y0, volume_usd, next_high, next_low, args.pool_liquidity, args.volume_multiplier
                )
                hourly_ret_pct = (v1 + fee_usd) / capital - 1.0  # gross of entry costs
                capital = v1 + fee_usd
                held_sol = x1
                prev_active = True

            net_hourly_ret_pct = capital / start_capital - 1.0

            records.append({
                "hour_idx": i - start_idx + 1,
                "datetime": raw_df.loc[candle_idx + 1, "datetime"],
                "current_price": current_price,
                "next_close": next_close,
                "pa": pa,
                "pb": pb,
                "pred_trend_coef": pred_trend_coef,
                "regime": regime,
                "status": status,
                "in_range": in_range,
                "fee_usd": fee_usd,
                "il_usd": il_usd,
                "hourly_return_pct": hourly_ret_pct,
                "net_hourly_return_pct": net_hourly_ret_pct,
                "capital": capital
            })

    df = pd.DataFrame(records)
    output_csv = os.path.join(base_dir, "backtest_results.csv")
    df.to_csv(output_csv, index=False)

    # Metrics Compilation
    active_mask = df["regime"].str.startswith("ACTIVE_LP")
    active_hours = int(active_mask.sum())
    cash_hours = len(df) - active_hours

    active_in_range_rate = (df.loc[active_mask, "status"] == "IN_RANGE").mean() * 100.0 if active_hours > 0 else 0.0
    total_in_range_rate = (df["status"] == "IN_RANGE").mean() * 100.0
    active_win_rate = (df.loc[active_mask, "net_hourly_return_pct"] > 0).mean() * 100.0 if active_hours > 0 else 0.0
    gross_win_rate = (df["net_hourly_return_pct"] > 0).mean() * 100.0

    total_fees = df["fee_usd"].sum()
    net_profit = capital - BASE_CAPITAL_USD

    # Annualized Sharpes
    hourly_returns = df["net_hourly_return_pct"]
    hourly_sharpe = (hourly_returns.mean() / (hourly_returns.std() + 1e-9)) * np.sqrt(8760)

    df["day"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("D")
    daily_returns = df.groupby("day")["net_hourly_return_pct"].sum()
    daily_sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(365)

    print("=======================================================")
    print(f"HISTORICAL BACKTEST SUMMARY ({TEST_HOURS} Held-out Test Hours)")
    print("=======================================================")
    print(f"Output CSV saved to:   {output_csv}")
    print(f"Active LP Hours:       {active_hours} | Cash Preservation Hours: {cash_hours}")
    print(f"In-Range Hit Rate (LP):{active_in_range_rate:.1f}% (Total: {total_in_range_rate:.1f}%)")
    print(f"Active Win Rate:       {active_win_rate:.1f}% (Gross: {gross_win_rate:.1f}%)")
    print(f"Total Accrued Fees:    ${total_fees:.2f}")
    print(f"Total Net Profit (ML): ${net_profit:.2f}")
    print(f"Hourly Sharpe:         {hourly_sharpe:.2f}")
    print(f"Daily Sharpe (365d):   {daily_sharpe:.2f}")
    print("=======================================================")

if __name__ == "__main__":
    main()