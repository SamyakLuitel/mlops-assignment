"""RPS sweep to find the SLO knee.

Runs load_test/driver.py at several offered RPS levels and tabulates the
metrics that actually matter under saturation: error rate and goodput
(successful requests/sec), alongside the P95 of successful requests.

The driver's percentiles are computed over successful requests only, so
P95 is only trustworthy when the error rate is near zero. This sweep
surfaces error rate next to it so you don't read a censored tail as a pass.

Run:
    uv run python load_test/sweep.py --rps 1,2,4,8 --duration 120

Find the highest RPS row where p95 < 5s AND err% ~ 0 -> that's your knee.
Then confirm it with a full 300s run at that RPS.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "load_test" / "driver.py"
SLO_P95_SECONDS = 5.0
SLO_MAX_ERR_RATE = 0.01  # treat >1% errors as "not a real pass"


def run_one(rps: float, duration: int, agent_url: str) -> dict:
    out = ROOT / "results" / f"sweep_rps{rps:g}.json"
    cmd = [
        sys.executable, str(DRIVER),
        "--rps", str(rps),
        "--duration", str(duration),
        "--agent-url", agent_url,
        "--out", str(out),
    ]
    print(f"\n=== offering {rps} RPS for {duration}s ===", flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())["summary"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rps", default="1,2,4,8", help="comma-separated RPS levels")
    p.add_argument("--duration", type=int, default=120, help="seconds per level")
    p.add_argument("--agent-url", default="http://localhost:8001/answer")
    args = p.parse_args()

    levels = [float(x) for x in args.rps.split(",") if x.strip()]
    rows = []
    for rps in levels:
        s = run_one(rps, args.duration, args.agent_url)
        total = s["total_requests"] or 1
        err_rate = (total - s["ok"]) / total
        goodput = s["ok"] / s["wall_clock_seconds"] if s["wall_clock_seconds"] else 0.0
        passes = s["latency_p95"] < SLO_P95_SECONDS and err_rate <= SLO_MAX_ERR_RATE
        rows.append({
            "rps": rps,
            "ok": s["ok"],
            "err_rate": err_rate,
            "goodput": goodput,
            "p50": s["latency_p50"],
            "p95": s["latency_p95"],
            "slo_pass": passes,
        })

    # Summary table.
    print("\n" + "=" * 78)
    print(f"{'offered':>8} {'ok':>5} {'err%':>7} {'goodput':>9} {'p50(s)':>9} {'p95(s)':>9} {'SLO<5s':>8}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['rps']:>8g} {r['ok']:>5d} {r['err_rate']*100:>6.1f}% "
            f"{r['goodput']:>9.2f} {r['p50']:>9.2f} {r['p95']:>9.2f} "
            f"{'PASS' if r['slo_pass'] else 'fail':>8}"
        )
    print("=" * 78)

    knees = [r for r in rows if r["slo_pass"]]
    if knees:
        best = max(knees, key=lambda r: r["rps"])
        print(f"\nKnee (highest passing RPS): {best['rps']:g} RPS "
              f"(p95 {best['p95']:.2f}s, {best['err_rate']*100:.1f}% errors).")
        print(f"Confirm with: uv run python load_test/driver.py --rps {best['rps']:g} --duration 300")
    else:
        print("\nNo level passed the SLO. Even the lowest RPS is over capacity or "
              "vLLM/agent is misconfigured - check the dashboard and which start_vllm script is running.")


if __name__ == "__main__":
    main()
