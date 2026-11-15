import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import CryptoTimeSeriesDataset
from model import OrcaCryptoTransformer
from config import (
    NUM_FEATURES, SEQ_LEN, SEED, CRITICAL_DROP_THRESHOLD, MODERATE_DROP_THRESHOLD, split_ranges,
    LOSS_Q_LOW, LOSS_Q_HIGH, LOSS_Q_TREND, LOSS_IL_PENALTY, LOSS_WIDTH_PENALTY, LOSS_TREND_WEIGHT,
    LOSS_CRASH_PENALTY, LOSS_DIRECTION_WEIGHT, LOSS_CRASH_TARGET,
)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4


class OrcaAsymmetricQuantileLoss(nn.Module):
    """
    Downside-Aware & Impermanent-Loss-Aware Multi-Task Quantile Loss:
    1. Pinball loss for high tick (q=0.90) and low tick (q=0.10)
    2. Asymmetric penalty for piercing the lower tick (il_penalty)
    3. Range width penalty to incentivize high capital concentration
    4. Calibrated asymmetric trend quantile loss (q=0.42, crash_penalty=2.0)
    5. Directional sign penalty centered at neutral 1.000

    Note: the IL penalty multiplies the loss when the low is pierced, so the effective low quantile is
    q_low / (q_low + (1 - q_low) * il_penalty) ~ 5% for the defaults, not q_low.
    """
    def __init__(
        self,
        q_low=LOSS_Q_LOW,
        q_high=LOSS_Q_HIGH,
        q_trend=LOSS_Q_TREND,
        il_penalty=LOSS_IL_PENALTY,
        width_penalty=LOSS_WIDTH_PENALTY,
        trend_weight=LOSS_TREND_WEIGHT,
        crash_penalty=LOSS_CRASH_PENALTY,
        direction_weight=LOSS_DIRECTION_WEIGHT,
        crash_target=LOSS_CRASH_TARGET,
    ):
        super().__init__()
        self.q_low = q_low
        self.q_high = q_high
        self.q_trend = q_trend
        self.il_penalty = il_penalty
        self.width_penalty = width_penalty
        self.trend_weight = trend_weight
        self.crash_penalty = crash_penalty
        self.direction_weight = direction_weight
        self.crash_target = crash_target

    def forward(self, preds, targets):
        pred_high, pred_low, pred_trend = preds[:, 0], preds[:, 1], preds[:, 2]
        target_high, target_low, target_trend = targets[:, 0], targets[:, 1], targets[:, 2]

        # 1. Upper Quantile (0.90) Pinball Loss
        err_high = target_high - pred_high
        loss_high = torch.max(self.q_high * err_high, (self.q_high - 1.0) * err_high)

        # 2. Lower Quantile (0.10) Pinball Loss + Asymmetric IL Penalty
        err_low = target_low - pred_low
        loss_low = torch.max(self.q_low * err_low, (self.q_low - 1.0) * err_low)
        il_mask = (target_low < pred_low).float()
        loss_low = loss_low * (1.0 + (self.il_penalty - 1.0) * il_mask)

        # 3. Range Width Regularizer (Incentivizes tight liquidity bands)
        spread = pred_high - pred_low
        loss_width = self.width_penalty * spread

        # 4. Calibrated Asymmetric Trend Loss (targets hourly drops below crash_target)
        err_trend = target_trend - pred_trend
        loss_trend_pinball = torch.max(
            self.q_trend * err_trend,
            (self.q_trend - 1.0) * err_trend,
        )
        crash_mask = ((target_trend < self.crash_target) & (pred_trend > target_trend)).float()
        loss_trend_pinball = loss_trend_pinball * (1.0 + (self.crash_penalty - 1.0) * crash_mask)

        # 5. Directional sign penalty centered at neutral 1.000
        center = 1.000
        dir_violation = F.relu(-(target_trend - center) * (pred_trend - center))
        loss_trend = self.trend_weight * loss_trend_pinball + self.direction_weight * dir_violation

        return torch.mean(loss_high + loss_low + loss_width + loss_trend)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true",
                    help="copy the best checkpoint to weights/orca_transformer.pth (existing file is backed up first)")
    ap.add_argument("--tag", default="", help="suffix for the run directory name")
    ap.add_argument("--loss", nargs="*", default=[], metavar="KEY=VAL",
                    help="override OrcaAsymmetricQuantileLoss kwargs, e.g. il_penalty=1.0 q_trend=0.5")
    args = ap.parse_args()
    loss_kwargs = {k: float(v) for k, v in (kv.split("=") for kv in args.loss)}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Initializing supervised fine-tuning on device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "sol_usdc_data.csv")

    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    dataset = CryptoTimeSeriesDataset(csv_file=csv_path, history_len=SEQ_LEN)
    splits = split_ranges(len(dataset))

    # Embargo between splits (EMBARGO_GAP = SEQ_LEN + 1 windows) removes temporal overlap.
    # The test split is reserved for backtest_historical.py and is never used here.
    train_indices = dataset.valid_indices(splits["train"])
    val_indices = dataset.valid_indices(splits["val"])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = OrcaCryptoTransformer(num_features=NUM_FEATURES, seq_len=SEQ_LEN).to(device)

    # Transfer pre-trained encoder weights
    pretrained_path = os.path.join(weights_dir, "orca_mae_pretrained.pth")
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        print(f"Pre-trained load: missing={len(missing)} unexpected={len(unexpected)}")
        print("Loaded pre-trained Masked Autoencoder weights. Starting supervised fine-tuning.")
    else:
        print("Pre-trained weights not found. Training from scratch.")

    criterion = OrcaAsymmetricQuantileLoss(**loss_kwargs).to(device)
    print(f"Loss overrides: {loss_kwargs or 'none (config defaults)'}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    run_dir = os.path.join(weights_dir, "runs", time.strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else ""))
    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, "orca_transformer.pth")

    print("\nStarting Downside-Aware & Impermanent-Loss-Aware Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            hours, days = batch[2].to(device), batch[3].to(device)

            optimizer.zero_grad()
            preds = model(x, hours, days)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # Validation evaluation
        model.eval()
        val_loss = 0.0
        total_cov = 0
        total_cash = 0
        total_skew = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                hours, days = batch[2].to(device), batch[3].to(device)

                preds = model(x, hours, days)
                loss = criterion(preds, y)
                val_loss += loss.item()

                pred_high, pred_low, pred_trend = preds[:, 0], preds[:, 1], preds[:, 2]
                target_high, target_low = y[:, 0], y[:, 1]

                # In-range coverage calculation
                covered = (target_low >= pred_low) & (target_high <= pred_high)
                total_cov += covered.sum().item()

                # Regime distribution tracking
                cash_mask = pred_trend < CRITICAL_DROP_THRESHOLD
                skew_mask = (pred_trend >= CRITICAL_DROP_THRESHOLD) & (pred_trend < MODERATE_DROP_THRESHOLD)
                total_cash += cash_mask.sum().item()
                total_skew += skew_mask.sum().item()
                total_samples += x.size(0)

        avg_val_loss = val_loss / len(val_loader)
        cov_pct = (total_cov / total_samples) * 100.0
        cash_pct = (total_cash / total_samples) * 100.0
        skew_pct = (total_skew / total_samples) * 100.0

        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = " [SAVED]"

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train: {avg_train_loss:.6f} | "
            f"Val: {avg_val_loss:.6f} | "
            f"Cov: {cov_pct:.1f}% | "
            f"Cash: {cash_pct:.1f}% | "
            f"Skew: {skew_pct:.1f}%"
            f"{saved_flag}"
        )

    print(f"\nTraining complete! Best checkpoint (val loss {best_val_loss:.6f}) saved to '{save_path}'.")
    prod_path = os.path.join(weights_dir, "orca_transformer.pth")
    if args.promote:
        if os.path.exists(prod_path):
            backup = f"{prod_path}.bak-{int(time.time())}"
            shutil.copy2(prod_path, backup)
            print(f"Previous production weights backed up to {backup}")
        shutil.copy2(save_path, prod_path)
        print(f"Promoted to production: {prod_path}")
    else:
        print("Production weights NOT changed. Evaluate with: python backtest_historical.py --weights <path>; "
              "re-run with --promote to deploy.")

if __name__ == "__main__":
    main()