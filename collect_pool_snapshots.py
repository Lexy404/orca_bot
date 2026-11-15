"""Hourly read-only snapshot of the Orca SOL/USDC whirlpool, appended to data/pool_snapshots.csv.

Purpose: build a real history of pool liquidity and fee accrual to replace the backtest's
--volume-multiplier assumption. Only reads public on-chain accounts: no keys, no transactions.

Per snapshot we store raw on-chain values (so nothing is lost) and a liquidity profile:
  - liquidity, sqrt_price, tick, fee_growth_global_a/b (raw integers)
  - profile_L: active liquidity (human sqrt units, same as backtest POOL_ACTIVE_LIQUIDITY) at tick offsets
    PROFILE_OFFSETS around the current tick (1 tick ~ 0.01% price), rebuilt from the neighbouring tick arrays.
Later analysis: the delta of fee_growth_global between two snapshots gives the fees per unit liquidity earned in
between, hence volume = fees / fee_rate (see analysis notes in CLAUDE.md).

Usage: python collect_pool_snapshots.py            # one snapshot (run hourly by launchd)
Env:   ORCA_POOL_ADDRESS, ORCA_SNAPSHOT_RPC (comma-separated RPC urls)
"""
import base64
import csv
import hashlib
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "data", "pool_snapshots.csv")

POOL = os.environ.get("ORCA_POOL_ADDRESS", "Czfq3xZZDmsdGdUyrNLtRhGc47cXcZtLG4crryfu44zE")
PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
RPCS = [u for u in os.environ.get(
    "ORCA_SNAPSHOT_RPC", "https://api.mainnet-beta.solana.com,https://solana-rpc.publicnode.com"
).split(",") if u]
TICK_ARRAY_SIZE = 88
TICK_BYTES = 113
DEC_A, DEC_B = 9, 6
L_SCALE = 10 ** ((DEC_A + DEC_B) / 2)  # raw liquidity -> human sqrt units
PROFILE_OFFSETS = [-200, -100, -50, -25, 0, 25, 50, 100, 200]  # ticks from the current tick

FIELDS = [
    "ts_utc", "slot", "tick_spacing", "fee_rate", "protocol_fee_rate", "tick", "price",
    "liquidity_raw", "sqrt_price_raw", "fee_growth_global_a", "fee_growth_global_b",
    "vault_sol", "vault_usdc", "profile_L",
]

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
P = 2**255 - 19
D = (-121665 * pow(121666, P - 2, P)) % P


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 58 + B58.index(ch)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return b"\0" * (len(s) - len(s.lstrip("1"))) + raw


def b58encode(b: bytes) -> str:
    n, out = int.from_bytes(b, "big"), ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + out


def on_curve(b: bytes) -> bool:
    y = int.from_bytes(b, "little") & ((1 << 255) - 1)
    if y >= P:
        return False
    num, den = (y * y - 1) % P, (D * y * y + 1) % P
    x2 = num * pow(den, P - 2, P) % P
    return x2 == 0 or pow(x2, (P - 1) // 2, P) == 1


def find_pda(seeds: list[bytes], program: str) -> str:
    prog = b58decode(program)
    for bump in range(255, -1, -1):
        h = hashlib.sha256(b"".join(seeds) + bytes([bump]) + prog + b"ProgramDerivedAddress").digest()
        if not on_curve(h):
            return b58encode(h)
    raise RuntimeError("no PDA found")


def rpc(method: str, params: list):
    last = None
    for url in RPCS:
        try:
            r = requests.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=20)
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(j["error"])
            return j["result"]
        except Exception as e:  # noqa: BLE001 - try the next endpoint
            last = e
    raise RuntimeError(f"all RPC endpoints failed: {last}")


def u128(d: bytes, off: int) -> int:
    lo, hi = struct.unpack_from("<QQ", d, off)
    return lo + (hi << 64)


def parse_pool(d: bytes) -> dict:
    return {
        "tick_spacing": struct.unpack_from("<H", d, 41)[0],
        "fee_rate": struct.unpack_from("<H", d, 45)[0],
        "protocol_fee_rate": struct.unpack_from("<H", d, 47)[0],
        "liquidity": u128(d, 49),
        "sqrt_price": u128(d, 65),
        "tick": struct.unpack_from("<i", d, 81)[0],
        "vault_a": b58encode(d[133:165]),
        "fee_growth_a": u128(d, 165),
        "vault_b": b58encode(d[213:245]),
        "fee_growth_b": u128(d, 245),
    }


def parse_tick_array(d: bytes, spacing: int) -> dict[int, int]:
    """{tick_index: liquidity_net} for the initialized ticks of one tick array account."""
    start = struct.unpack_from("<i", d, 8)[0]
    out = {}
    for i in range(TICK_ARRAY_SIZE):
        off = 12 + i * TICK_BYTES
        if d[off] == 1:
            lo, hi = struct.unpack_from("<Qq", d, off + 1)  # liquidity_net is i128
            out[start + i * spacing] = lo + (hi << 64)
    return out


def tick_array_start(tick: int, spacing: int) -> int:
    span = spacing * TICK_ARRAY_SIZE
    return (tick // span) * span  # floor division: correct for negative ticks


def liquidity_profile(pool: dict) -> dict[int, float]:
    """Active liquidity (human units) at PROFILE_OFFSETS ticks around the current tick."""
    spacing, tick = pool["tick_spacing"], pool["tick"]
    span = spacing * TICK_ARRAY_SIZE
    base = tick_array_start(tick, spacing)
    nets: dict[int, int] = {}
    for k in (-1, 0, 1):
        pda = find_pda([b"tick_array", b58decode(POOL), str(base + k * span).encode()], PROGRAM)
        acc = rpc("getAccountInfo", [pda, {"encoding": "base64"}])["value"]
        if acc:  # a neighbouring uninitialized tick array simply has no liquidity changes
            nets.update(parse_tick_array(base64.b64decode(acc["data"][0]), spacing))
        elif k == 0:  # the array holding the current tick must exist; otherwise the PDA derivation is wrong
            raise RuntimeError(f"current tick array {pda} (start {base}) not found")

    ticks = sorted(nets)
    profile = {}
    for off in PROFILE_OFFSETS:
        target = tick + off
        liq = pool["liquidity"]
        if target >= tick:   # crossing initialized ticks upwards adds their net liquidity
            liq += sum(nets[t] for t in ticks if tick < t <= target)
        else:                # crossing downwards removes it (a tick at `tick` itself is already below price)
            liq -= sum(nets[t] for t in ticks if target < t <= tick)
        profile[off] = max(liq, 0) / L_SCALE
    return profile


def snapshot() -> dict:
    res = rpc("getAccountInfo", [POOL, {"encoding": "base64"}])
    d = base64.b64decode(res["value"]["data"][0])
    if len(d) < 261 or res["value"]["owner"] != PROGRAM:
        raise RuntimeError("unexpected pool account (wrong owner or layout)")
    pool = parse_pool(d)
    price = (pool["sqrt_price"] / 2**64) ** 2 * 10 ** (DEC_A - DEC_B)
    vault_sol = float(rpc("getTokenAccountBalance", [pool["vault_a"]])["value"]["uiAmount"])
    vault_usdc = float(rpc("getTokenAccountBalance", [pool["vault_b"]])["value"]["uiAmount"])
    profile = liquidity_profile(pool)
    return {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "slot": res["context"]["slot"],
        "tick_spacing": pool["tick_spacing"],
        "fee_rate": pool["fee_rate"],
        "protocol_fee_rate": pool["protocol_fee_rate"],
        "tick": pool["tick"],
        "price": round(price, 6),
        "liquidity_raw": pool["liquidity"],
        "sqrt_price_raw": pool["sqrt_price"],
        "fee_growth_global_a": pool["fee_growth_a"],
        "fee_growth_global_b": pool["fee_growth_b"],
        "vault_sol": vault_sol,
        "vault_usdc": vault_usdc,
        "profile_L": json.dumps({str(k): round(v, 1) for k, v in profile.items()}),
    }


def main() -> int:
    try:
        row = snapshot()
    except Exception as e:  # noqa: BLE001
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] snapshot failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{row['ts_utc']}] saved snapshot: tick={row['tick']} price={row['price']} L_raw={row['liquidity_raw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
