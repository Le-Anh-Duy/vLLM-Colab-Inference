#!/usr/bin/env python3
"""Mo phong traffic goi vLLM OpenAI-compatible server va cham diem theo cong
thuc ERS (Effective Request Score) trong Statement.txt (track 3, Viettel AI
Race 2026).

Chay tren Colab (can `pip install httpx` neu chua co):
    python3 scripts/benchmark_traffic.py --synthetic --rps 2 --duration 30

Hoac voi trace file JSONL co san (moi dong 1 request):
    {"id": "0", "arrival_time": 0.0, "prompt": "...", "max_tokens": 128}
    python3 scripts/benchmark_traffic.py --trace-file trace.jsonl

Cac nguong Floor/Ceiling/w/gamma la PLACEHOLDER - BTC se cong bo gia tri
chinh thuc theo tung vong, luc do chi can doi qua CLI flag, khong sua code.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("Thieu thu vien httpx. Chay: pip install httpx", file=sys.stderr)
    raise


# --------------------------------------------------------------------------
# Trace: doc tu file JSONL hoac sinh Poisson process gia lap
# --------------------------------------------------------------------------

SAMPLE_PROMPTS = [
    "Hay giai thich ngan gon ve dinh luat bao toan nang luong.",
    "Viet mot doan van gioi thieu ve thanh pho Ha Noi.",
    "Tom tat uu nhuoc diem cua kien truc Transformer trong 3 cau.",
    "Cho vi du ve mot thuat toan sap xep va do phuc tap cua no.",
    "Giai thich su khac biet giua TTFT va TPOT trong LLM serving.",
    "Liet ke 5 buoc de toi uu hoa mot he thong inference LLM.",
    "Mo ta ngan gon co che Paged Attention hoat dong nhu the nao.",
    "Tai sao KV cache quantization giup tiet kiem VRAM?",
]


@dataclass
class TraceRequest:
    id: str
    arrival_time: float  # giay, tinh tu t0
    prompt: str
    max_tokens: int = 128


def load_trace_file(path: str) -> list[TraceRequest]:
    requests: list[TraceRequest] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            requests.append(
                TraceRequest(
                    id=str(row.get("id", i)),
                    arrival_time=float(row["arrival_time"]),
                    prompt=row.get("prompt") or random.choice(SAMPLE_PROMPTS),
                    max_tokens=int(row.get("max_tokens", 128)),
                )
            )
    requests.sort(key=lambda r: r.arrival_time)
    return requests


def generate_synthetic_trace(
    rps: float, duration: float, max_tokens_range: tuple[int, int], seed: int
) -> list[TraceRequest]:
    """Poisson arrival process voi rate co dinh `rps` trong `duration` giay."""
    rng = random.Random(seed)
    requests: list[TraceRequest] = []
    t = 0.0
    i = 0
    while t < duration:
        t += rng.expovariate(rps)
        if t >= duration:
            break
        requests.append(
            TraceRequest(
                id=str(i),
                arrival_time=t,
                prompt=rng.choice(SAMPLE_PROMPTS),
                max_tokens=rng.randint(*max_tokens_range),
            )
        )
        i += 1
    return requests


# --------------------------------------------------------------------------
# Goi request toi server, do TTFT/TPOT theo streaming chat completion
# --------------------------------------------------------------------------


@dataclass
class RequestResult:
    id: str
    success: bool = False
    error: str | None = None
    ttft: float | None = None  # giay
    tpot: float | None = None  # giay/token
    num_output_tokens: int = 0
    total_latency: float | None = None


async def run_one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    req: TraceRequest,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    result = RequestResult(id=req.id)
    send_time = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float | None = None
    token_count = 0

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                result.error = f"HTTP {resp.status_code}: {body[:200]!r}"
                return result

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
                    token_count += 1
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    end_time = time.perf_counter()
    if token_count == 0 or first_token_time is None:
        result.error = "khong nhan duoc token nao (0 token / timeout)"
        return result

    result.success = True
    result.ttft = first_token_time - send_time
    result.num_output_tokens = token_count
    result.total_latency = end_time - send_time
    if token_count > 1:
        result.tpot = (last_token_time - first_token_time) / (token_count - 1)
    else:
        result.tpot = 0.0
    return result


async def run_trace(
    requests: list[TraceRequest],
    base_url: str,
    model: str,
    timeout: float,
    max_concurrency: int,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[RequestResult | None] = [None] * len(requests)
    t0 = time.perf_counter()

    async def schedule_and_run(idx: int, req: TraceRequest) -> None:
        delay = req.arrival_time - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            results[idx] = await run_one_request(client, base_url, model, req, timeout)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *(schedule_and_run(i, r) for i, r in enumerate(requests))
        )

    return [r for r in results if r is not None]


# --------------------------------------------------------------------------
# ERS scoring - cong thuc lay tu Statement.txt muc 3.2
# --------------------------------------------------------------------------


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_component(value: float, floor: float, ceiling: float, gamma: float) -> float:
    if ceiling <= floor:
        return 1.0 if value <= floor else 0.0
    frac = clamp01((ceiling - value) / (ceiling - floor))
    return frac**gamma


def score_request(
    result: RequestResult,
    floor_ttft: float,
    ceil_ttft: float,
    floor_tpot: float,
    ceil_tpot: float,
    w: float,
    gamma: float,
) -> float:
    if not result.success or result.num_output_tokens == 0:
        return 0.0
    s_ttft = score_component(result.ttft, floor_ttft, ceil_ttft, gamma)
    s_tpot = score_component(result.tpot, floor_tpot, ceil_tpot, gamma)
    return w * s_ttft + (1 - w) * s_tpot


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def summarize(
    results: list[RequestResult],
    floor_ttft: float,
    ceil_ttft: float,
    floor_tpot: float,
    ceil_tpot: float,
    w: float,
    gamma: float,
) -> dict[str, Any]:
    scores = [
        score_request(r, floor_ttft, ceil_ttft, floor_tpot, ceil_tpot, w, gamma)
        for r in results
    ]
    ok = [r for r in results if r.success]
    ttfts = [r.ttft for r in ok]
    tpots = [r.tpot for r in ok]

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
        return {
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    return {
        "num_requests": len(results),
        "num_success": len(ok),
        "num_failed": len(results) - len(ok),
        "ers": statistics.mean(scores) if scores else 0.0,
        "ttft_sec": stats(ttfts),
        "tpot_sec": stats(tpots),
        "thresholds": {
            "floor_ttft": floor_ttft,
            "ceil_ttft": ceil_ttft,
            "floor_tpot": floor_tpot,
            "ceil_tpot": ceil_tpot,
            "w": w,
            "gamma": gamma,
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000", help="Goc URL cua vLLM server")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Ten model (phai khop --model luc serve)")
    p.add_argument("--trace-file", default=None, help="Duong dan file JSONL trace co san")
    p.add_argument("--synthetic", action="store_true", help="Sinh trace Poisson gia lap thay vi doc file")
    p.add_argument("--rps", type=float, default=2.0, help="Request/giay cho trace gia lap")
    p.add_argument("--duration", type=float, default=30.0, help="Thoi luong trace gia lap (giay)")
    p.add_argument("--max-tokens-min", type=int, default=32)
    p.add_argument("--max-tokens-max", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=60.0, help="Timeout moi request (giay)")
    p.add_argument("--max-concurrency", type=int, default=64, help="Gioi han so request dong thoi")
    # Nguong ERS - PLACEHOLDER, doi khi BTC cong bo gia tri chinh thuc tung vong
    p.add_argument("--floor-ttft", type=float, default=0.3, help="Floor TTFT (giay)")
    p.add_argument("--ceil-ttft", type=float, default=3.0, help="Ceiling TTFT (giay)")
    p.add_argument("--floor-tpot", type=float, default=0.05, help="Floor TPOT (giay/token)")
    p.add_argument("--ceil-tpot", type=float, default=0.2, help="Ceiling TPOT (giay/token)")
    p.add_argument("--w", type=float, default=0.5, help="Trong so TTFT (0<w<1)")
    p.add_argument("--gamma", type=float, default=1.0, help="He so luy thua penalty curve (>=1)")
    p.add_argument("--output-dir", default="benchmark_results", help="Thu muc ghi ket qua")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.trace_file:
        requests = load_trace_file(args.trace_file)
    elif args.synthetic:
        requests = generate_synthetic_trace(
            rps=args.rps,
            duration=args.duration,
            max_tokens_range=(args.max_tokens_min, args.max_tokens_max),
            seed=args.seed,
        )
    else:
        print("Can chi dinh --trace-file hoac --synthetic", file=sys.stderr)
        sys.exit(1)

    print(f"==> {len(requests)} request se duoc gui toi {args.base_url} (model={args.model})")

    results = asyncio.run(
        run_trace(
            requests,
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
        )
    )

    summary = summarize(
        results,
        floor_ttft=args.floor_ttft,
        ceil_ttft=args.ceil_ttft,
        floor_tpot=args.floor_tpot,
        ceil_tpot=args.ceil_tpot,
        w=args.w,
        gamma=args.gamma,
    )

    print("========================================")
    print(f"So request           : {summary['num_requests']}")
    print(f"Thanh cong / That bai : {summary['num_success']} / {summary['num_failed']}")
    print(f"TTFT (mean/p50/p95)  : {summary['ttft_sec']['mean']:.3f}s / {summary['ttft_sec']['p50']:.3f}s / {summary['ttft_sec']['p95']:.3f}s")
    print(f"TPOT (mean/p50/p95)  : {summary['tpot_sec']['mean']:.3f}s / {summary['tpot_sec']['p50']:.3f}s / {summary['tpot_sec']['p95']:.3f}s")
    print(f"ERS (uoc luong)      : {summary['ers']:.4f}")
    print("(Nguong Floor/Ceiling/w/gamma la PLACEHOLDER, doi qua CLI flag khi BTC cong bo gia tri chinh thuc)")
    print("========================================")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "per_request.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "success", "error", "ttft_sec", "tpot_sec", "num_output_tokens", "total_latency_sec"])
        for r in results:
            writer.writerow([r.id, r.success, r.error or "", r.ttft, r.tpot, r.num_output_tokens, r.total_latency])

    print(f"==> Da ghi ket qua vao {out_dir}/summary.json va {out_dir}/per_request.csv")


if __name__ == "__main__":
    main()
