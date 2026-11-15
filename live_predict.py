import os
import json
import time
import zmq
import torch
import numpy as np
from dataset import build_single_window_features
from model import OrcaCryptoTransformer
from config import (
    NUM_FEATURES,
    ZMQ_PORT_DEFAULT,
    MIN_LIVE_CANDLES,
    MAX_LIVE_CANDLES,
    MAX_PRICE_DEVIATION,
    MAX_CANDLE_AGE_SEC,
    validate_bounds,
    CRITICAL_DROP_THRESHOLD,
    MODERATE_DROP_THRESHOLD,
    skewed_bounds,
)

ZMQ_PORT = ZMQ_PORT_DEFAULT
ZMQ_BIND_HOST = os.environ.get("ORCA_ZMQ_HOST", "127.0.0.1")  # local executor only; set explicitly to expose
ALLOW_STALE = os.environ.get("ORCA_ALLOW_STALE") == "1"  # testing only: skip the candle-freshness check


def error_response(message: str) -> dict:
    return {"status": "error", "message": message}


def handle_request(model, device, message_str: str, now: float | None = None) -> dict:
    """Parses a request and returns a response dict. Never raises for bad input."""
    try:
        payload = json.loads(message_str)
        if not isinstance(payload, dict):
            return error_response("Request must be a JSON object.")

        candles = payload.get("candles")
        if not isinstance(candles, list):
            return error_response("'candles' must be a list.")
        if len(candles) < MIN_LIVE_CANDLES:
            return error_response(
                f"Insufficient candle history. Received {len(candles)}, required >= {MIN_LIVE_CANDLES}."
            )
        if len(candles) > MAX_LIVE_CANDLES:
            return error_response(f"Too many candles. Received {len(candles)}, max {MAX_LIVE_CANDLES}.")

        if "current_price" not in payload:
            return error_response("Missing 'current_price'.")
        current_price = float(payload["current_price"])
        if not np.isfinite(current_price) or current_price <= 0:
            return error_response(f"Invalid current_price: {current_price}")

        x_np, hours_list, days_list, base_price = build_single_window_features(candles)
        if abs(current_price / base_price - 1.0) > MAX_PRICE_DEVIATION:
            return error_response(
                f"current_price {current_price} deviates >{MAX_PRICE_DEVIATION:.0%} from last candle close {base_price}"
            )

        if not ALLOW_STALE:
            age = (time.time() if now is None else now) - int(candles[-1]["time"])
            if age > MAX_CANDLE_AGE_SEC:
                return error_response(f"Stale candles: last candle opened {age / 3600:.1f}h ago (max {MAX_CANDLE_AGE_SEC / 3600:.0f}h).")

        x_tensor = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(device)
        hours_tensor = torch.tensor(hours_list, dtype=torch.long).unsqueeze(0).to(device)
        days_tensor = torch.tensor(days_list, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(x_tensor, hours_tensor, days_tensor)[0].cpu().numpy()

        if not np.isfinite(preds).all():
            return error_response("Model produced non-finite output.")

        pred_high_coef, pred_low_coef, pred_trend_coef = (float(v) for v in preds)

        # Coefficients are relative to the last candle close (the model's base price).
        raw_pred_high = base_price * pred_high_coef
        raw_pred_low = base_price * pred_low_coef
        spread = raw_pred_high - raw_pred_low

        if pred_trend_coef < CRITICAL_DROP_THRESHOLD:
            regime = "SKIPPED_CASH"
            skip_hour = True
            pred_low = current_price
            pred_high = current_price
        elif pred_trend_coef < MODERATE_DROP_THRESHOLD:
            regime = "SKEW_RANGE_DOWN"
            skip_hour = False
            pred_low, pred_high = skewed_bounds(base_price, raw_pred_low, raw_pred_high)
        else:
            regime = "NORMAL_LP"
            skip_hour = False
            pred_low = raw_pred_low
            pred_high = raw_pred_high

        if regime != "SKIPPED_CASH":
            validate_bounds(pred_low, pred_high, current_price)

        return {
            "status": "success",
            "regime": regime,
            "skip_hour": skip_hour,
            "current_price": current_price,
            "pred_low": round(float(pred_low), 4),
            "pred_high": round(float(pred_high), 4),
            "pred_trend_coef": round(pred_trend_coef, 6),
            "raw_spread_pct": round(float(spread / base_price * 100.0), 3),
            "price_in_range": bool(pred_low <= current_price <= pred_high),
        }
    except Exception as e:  # noqa: BLE001 - every failure must still yield a JSON reply
        return error_response(f"{type(e).__name__}: {e}")


def safe_dumps(response: dict) -> str:
    try:
        return json.dumps(response, allow_nan=False)
    except (ValueError, TypeError) as e:
        return json.dumps(error_response(f"Response serialization failed: {e}"))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Initializing ZeroMQ Inference Server on device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "weights", "orca_transformer.pth")

    model = OrcaCryptoTransformer(num_features=NUM_FEATURES).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded production weights from: {weights_path}")

    context = zmq.Context()

    def make_socket():
        sock = context.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, 5000)
        sock.bind(f"tcp://{ZMQ_BIND_HOST}:{ZMQ_PORT}")
        return sock

    socket = make_socket()
    print(f"Orca Alpha Engine listening on tcp://{ZMQ_BIND_HOST}:{ZMQ_PORT}...")

    def reset_socket():
        # REP is stuck in the wrong state (EFSM) -> the only safe recovery is a fresh socket.
        nonlocal socket
        socket.close(linger=0)
        socket = make_socket()

    while True:
        try:
            raw = socket.recv()
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break
            print(f"recv failed: {e}")
            reset_socket()
            continue

        try:
            response = handle_request(model, device, raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            response = error_response(f"{type(e).__name__}: {e}")

        try:
            socket.send_string(safe_dumps(response))
        except zmq.ZMQError as e:
            print(f"send failed: {e}")
            reset_socket()


if __name__ == "__main__":
    main()
