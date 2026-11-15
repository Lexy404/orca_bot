import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_FEATURES, SEQ_LEN, MIN_SPREAD, LOW_OFFSET_SCALE, SPREAD_SCALE, MAX_TREND_DEV


class OrcaCryptoTransformer(nn.Module):
    """
    Pre-LN Transformer Encoder for Solana CLMM Liquidity Range & Trend Prediction.
    Consumes NUM_FEATURES (=11) engineered features per candle:
      0: open, 1: high, 2: low, 3: close, 4: volume,
      5: log_return, 6: hl_range_pct, 7: realized_vol_6h,
      8: natr_14 (gap-aware Normalized ATR), 9: chaikin_vol (Chaikin Volatility ROC),
      10: clv (Close Location Value in [-1, 1])
    Guarantees non-crossing quantile intervals via monotonic softplus parameterization:
      - pred_low_coef = 1.0 - softplus(raw_low_offset) * LOW_OFFSET_SCALE
      - pred_high_coef = pred_low_coef + min_spread + softplus(raw_spread) * SPREAD_SCALE
      - pred_trend_coef = 1.0 + tanh(raw_trend) * MAX_TREND_DEV  (all constants in config.py)
    """

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        seq_len: int = SEQ_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        min_spread: float = MIN_SPREAD,
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.min_spread = min_spread

        self.price_embedding = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.hour_embedding = nn.Embedding(24, d_model)
        self.day_embedding = nn.Embedding(7, d_model)

        self.pos_embedding = nn.Embedding(seq_len + 1, d_model)
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Pre-LN (norm_first=True) with enable_nested_tensor=False to suppress PyTorch warnings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(d_model * 2)

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )

    def encode_sequence(self, x: torch.Tensor, hours: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x_price = self.price_embedding(x)
        x_hour = self.hour_embedding(hours)
        x_day = self.day_embedding(days)
        tokens = x_price + x_hour + x_day

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_combined = torch.cat((cls_tokens, tokens), dim=1)

        positions = torch.arange(0, seq_len + 1, device=x.device).unsqueeze(0)
        x_combined = x_combined + self.pos_embedding(positions)
        x_combined = self.dropout(x_combined)

        return self.transformer_encoder(x_combined)

    def forward(self, x: torch.Tensor, hours: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        transformer_out = self.encode_sequence(x, hours, days)

        cls_out = transformer_out[:, 0, :]
        last_out = transformer_out[:, -1, :]

        combined_context = self.final_norm(torch.cat((cls_out, last_out), dim=-1))
        raw_out = self.prediction_head(combined_context)

        # Parameterize strictly valid non-crossing CLMM bounds around 1.0 (current spot price)
        low_offset = F.softplus(raw_out[:, 0]) * LOW_OFFSET_SCALE
        pred_low = 1.0 - low_offset

        spread = self.min_spread + F.softplus(raw_out[:, 1]) * SPREAD_SCALE
        pred_high = pred_low + spread

        pred_trend = 1.0 + torch.tanh(raw_out[:, 2]) * MAX_TREND_DEV

        return torch.stack([pred_high, pred_low, pred_trend], dim=-1)


class OrcaMaskedAutoencoder(nn.Module):
    """
    Self-supervised Masked Autoencoder (MAE) wrapper for pre-training OrcaCryptoTransformer
    via contiguous block masking across all NUM_FEATURES input features.
    """

    def __init__(self, base_model: OrcaCryptoTransformer):
        super().__init__()
        self.encoder = base_model
        d_model = base_model.d_model
        num_features = base_model.num_features

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        hours: torch.Tensor,
        days: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x_price = self.encoder.price_embedding(x)

        expanded_mask = mask.unsqueeze(-1).expand_as(x_price)
        x_price = torch.where(expanded_mask, self.mask_token, x_price)

        x_hour = self.encoder.hour_embedding(hours)
        x_day = self.encoder.day_embedding(days)
        tokens = x_price + x_hour + x_day

        cls_tokens = self.encoder.cls_token.expand(batch_size, -1, -1)
        x_combined = torch.cat((cls_tokens, tokens), dim=1)

        positions = torch.arange(0, seq_len + 1, device=x.device).unsqueeze(0)
        x_combined = x_combined + self.encoder.pos_embedding(positions)
        x_combined = self.encoder.dropout(x_combined)

        transformer_out = self.encoder.transformer_encoder(x_combined)

        sequence_out = transformer_out[:, 1:, :]
        reconstructed_x = self.reconstruction_head(sequence_out)

        return reconstructed_x