import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import CryptoTimeSeriesDataset
from model import OrcaCryptoTransformer, OrcaMaskedAutoencoder
from config import NUM_FEATURES, SEQ_LEN, SEED, split_ranges

# Configuration
MASK_RATIO = 0.20
BLOCK_SIZE = 4
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4

def make_block_mask(batch_size: int, seq_len: int, mask_ratio: float, block_size: int, device) -> torch.Tensor:
    """Contiguous block masking: bool [batch, seq_len], True = masked."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    num_blocks = max(1, int(round(mask_ratio * seq_len / block_size)))
    starts = torch.randint(0, seq_len - block_size + 1, (batch_size, num_blocks), device=device)
    for b in range(block_size):
        mask.scatter_(1, starts + b, True)
    return mask


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Initializing self-supervised pre-training on device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "sol_usdc_data.csv")

    weights_dir = os.path.join(base_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    dataset = CryptoTimeSeriesDataset(csv_file=csv_path, history_len=SEQ_LEN)
    splits = split_ranges(len(dataset))

    # Isolation: train encoder strictly on the train split; same split as train.py
    train_dataset = Subset(dataset, dataset.valid_indices(splits["train"]))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    base_model = OrcaCryptoTransformer(num_features=NUM_FEATURES, seq_len=SEQ_LEN)
    model = OrcaMaskedAutoencoder(base_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.HuberLoss()

    print(f"Starting Masked Autoencoder Pre-training on {len(train_dataset)} windows...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)
            hours = batch[2].to(device)
            days = batch[3].to(device)
            mask = make_block_mask(x.size(0), x.size(1), MASK_RATIO, BLOCK_SIZE, device)
            optimizer.zero_grad()
            recon = model(x, hours, days, mask)
            loss = criterion(recon[mask], x[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Pre-training Huber Loss: {avg_loss:.6f}")

    pretrained_path = os.path.join(weights_dir, "orca_mae_pretrained.pth")
    if os.path.exists(pretrained_path):
        backup = f"{pretrained_path}.bak-{int(time.time())}"
        shutil.copy2(pretrained_path, backup)
        print(f"Previous pre-trained weights backed up to {backup}")
    torch.save(model.encoder.state_dict(), pretrained_path)
    print(f"Pre-training complete! Encoder weights saved to: '{pretrained_path}'.")

if __name__ == "__main__":
    main()