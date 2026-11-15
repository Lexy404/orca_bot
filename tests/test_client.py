import json
import os
import sys

import pandas as pd
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))

from config import MIN_LIVE_CANDLES, ZMQ_PORT_DEFAULT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ping_ai():
    """Sends the last MIN_LIVE_CANDLES candles of the local CSV as a protocol-conformant request."""
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "sol_usdc_data.csv")).tail(MIN_LIVE_CANDLES)
    candles = [
        {"time": int(r.timestamp), "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "volume": float(r.volume)}
        for r in df.itertuples()
    ]
    request = {"current_price": candles[-1]["close"], "candles": candles}

    print("Note: the server rejects stale candles; start it with ORCA_ALLOW_STALE=1 to test with the local CSV.")
    print("Connecting to AI Inference Engine over ZeroMQ...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT_DEFAULT}")

    socket.send_string(json.dumps(request))
    response = socket.recv_json()
    print("\nReceived prediction payload:")
    print(json.dumps(response, indent=4))


if __name__ == "__main__":
    ping_ai()
