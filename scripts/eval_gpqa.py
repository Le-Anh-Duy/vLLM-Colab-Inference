#!/usr/bin/env python3
"""Cham accuracy GPQA Diamond cho server vLLM dang chay, phuc vu Accuracy
Gate trong Statement.txt (Viettel AI Race 2026, vong 1 so loai, muc 2).

BTC dung 100 cau hoi CO DINH trich tu GPQA Diamond (chua cong bo o thoi diem
viet script nay). Script nay nhan file cau hoi qua --questions-file (JSONL) -
khi BTC phat hanh file that, chi can tro --questions-file vao do, KHONG can
sua code.

Dinh dang JSONL mong doi, moi dong 1 cau hoi:
    {
      "id": "0",
      "question": "...",
      "choices": ["dap an A", "dap an B", "dap an C", "dap an D"],
      "answer_index": 2          // hoac "answer": "C"
    }

Chay tren Colab (can `pip install httpx` neu chua co):
    python3 scripts/eval_gpqa.py --questions-file gpqa_diamond.jsonl

De lay bo cau hoi public GPQA Diamond tu HuggingFace (gated, can chap nhan
dieu khoan + HF_TOKEN, CHI dung de tu test - khong phai bo BTC se cham):
    pip install datasets
    python3 scripts/eval_gpqa.py --hf-fallback --num-questions 100
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("Thieu thu vien httpx. Chay: pip install httpx", file=sys.stderr)
    raise

from scoring import accuracy_decay

LETTERS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = """Answer the following multiple choice question. Think step \
by step, then finish your response with EXACTLY one line in the format \
"Answer: X" where X is one of A, B, C, D.

{question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}"""

ANSWER_RE = re.compile(r"Answer:\s*\(?([A-D])\)?", re.IGNORECASE)


@dataclass
class GpqaQuestion:
    id: str
    question: str
    choices: list[str]  # dung 4 phan tu, dap an dung o correct_index
    correct_index: int


def _normalize_row(row: dict[str, Any], idx: int) -> GpqaQuestion:
    choices = row.get("choices")
    if not choices or len(choices) != 4:
        raise ValueError(f"Dong {idx}: can dung 4 'choices', nhan duoc {choices!r}")

    if "answer_index" in row:
        correct_index = int(row["answer_index"])
    elif "answer" in row:
        letter = str(row["answer"]).strip().upper()
        if letter not in LETTERS:
            raise ValueError(f"Dong {idx}: 'answer'='{letter}' khong hop le (can A-D)")
        correct_index = LETTERS.index(letter)
    else:
        raise ValueError(f"Dong {idx}: thieu 'answer_index' hoac 'answer'")

    return GpqaQuestion(
        id=str(row.get("id", idx)),
        question=row["question"],
        choices=list(choices),
        correct_index=correct_index,
    )


def load_questions_file(path: str) -> list[GpqaQuestion]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            questions.append(_normalize_row(json.loads(line), idx))
    return questions


def load_from_hf(num_questions: int, seed: int) -> list[GpqaQuestion]:
    """Tai bo GPQA Diamond PUBLIC tu HuggingFace de tu test. Dataset nay
    gated - can `huggingface-cli login` / bien moi truong HF_TOKEN va chap
    nhan dieu khoan truy cap truoc. Day KHONG phai bo cau hoi BTC dung de
    cham diem that."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Thieu thu vien datasets. Chay: pip install datasets", file=sys.stderr)
        raise

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:num_questions]

    questions = []
    for i in indices:
        row = ds[i]
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Xao vi tri de tranh thien vi vi tri (giong cach eval GPQA chuan),
        # seed rieng cho tung cau de reproducible.
        order = list(range(4))
        random.Random(f"{seed}-{i}").shuffle(order)
        shuffled = [options[j] for j in order]
        correct_index = order.index(0)
        questions.append(
            GpqaQuestion(
                id=str(row.get("Record ID", i)),
                question=row["Question"],
                choices=shuffled,
                correct_index=correct_index,
            )
        )
    return questions


async def ask_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    q: GpqaQuestion,
    max_tokens: int,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        question=q.question,
        choice_a=q.choices[0],
        choice_b=q.choices[1],
        choice_c=q.choices[2],
        choice_d=q.choices[3],
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    async with semaphore:
        try:
            resp = await client.post(
                f"{base_url}/v1/chat/completions", json=payload, timeout=timeout
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            return {
                "id": q.id,
                "correct": False,
                "predicted_letter": None,
                "correct_letter": LETTERS[q.correct_index],
                "error": f"{type(exc).__name__}: {exc}",
                "raw_output": "",
            }

    match = ANSWER_RE.search(content)
    predicted_letter = match.group(1).upper() if match else None
    correct_letter = LETTERS[q.correct_index]

    return {
        "id": q.id,
        "correct": predicted_letter == correct_letter,
        "predicted_letter": predicted_letter,
        "correct_letter": correct_letter,
        "error": None if match else "khong parse duoc dap an tu output",
        "raw_output": content[-500:],  # cat bot cho gon file CSV
    }


async def run_eval(
    questions: list[GpqaQuestion],
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *(
                ask_one(client, base_url, model, q, max_tokens, timeout, semaphore)
                for q in questions
            )
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--questions-file", default=None, help="File JSONL cau hoi (uu tien dung cai nay khi BTC phat hanh)")
    p.add_argument("--hf-fallback", action="store_true", help="Tai GPQA Diamond public tu HuggingFace de tu test (can 'datasets' + quyen truy cap)")
    p.add_argument("--num-questions", type=int, default=100, help="Chi dung voi --hf-fallback")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024, help="Du cho CoT reasoning truoc khi ra 'Answer: X'")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--baseline-accuracy", type=float, default=0.4, help="baseline_accuracy theo thang 0..1 (Statement.txt vong 1: mac dinh 0.4), dung de tinh f(delta)")
    p.add_argument("--output-dir", default="gpqa_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.questions_file:
        questions = load_questions_file(args.questions_file)
    elif args.hf_fallback:
        questions = load_from_hf(args.num_questions, args.seed)
    else:
        print("Can chi dinh --questions-file hoac --hf-fallback", file=sys.stderr)
        sys.exit(1)

    print(f"==> Cham {len(questions)} cau hoi GPQA tren {args.base_url} (model={args.model})")

    results = asyncio.run(
        run_eval(
            questions,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
        )
    )

    num_correct = sum(1 for r in results if r["correct"])
    num_unparsed = sum(1 for r in results if r["predicted_letter"] is None)
    # accuracy theo thang 0..1, dung truc tiep voi accuracy_decay() va
    # baseline_accuracy (Statement.txt dung thang nay, vd baseline=0.4=40%).
    accuracy = num_correct / len(results) if results else 0.0

    print("========================================")
    print(f"So cau hoi         : {len(results)}")
    print(f"So cau dung        : {num_correct}")
    print(f"So cau khong parse duoc dap an : {num_unparsed}")
    print(f"Accuracy           : {accuracy:.4f} ({accuracy * 100:.2f}%)")

    summary: dict[str, Any] = {
        "num_questions": len(results),
        "num_correct": num_correct,
        "num_unparsed": num_unparsed,
        "accuracy": accuracy,
    }

    if args.baseline_accuracy is not None:
        delta = args.baseline_accuracy - accuracy
        penalty = accuracy_decay(delta)
        summary["baseline_accuracy"] = args.baseline_accuracy
        summary["delta"] = delta
        summary["accuracy_penalty_f_delta"] = penalty
        print(f"Baseline accuracy  : {args.baseline_accuracy:.4f} ({args.baseline_accuracy * 100:.2f}%)")
        print(f"Delta              : {delta:.4f}")
        print(f"f(delta) (penalty) : {penalty:.4f}")
    else:
        print("(Truyen --baseline-accuracy de tinh f(delta) theo Statement.txt muc 2)")
    print("========================================")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "per_question.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "correct", "predicted_letter", "correct_letter", "error", "raw_output"])
        for r in results:
            writer.writerow(
                [r["id"], r["correct"], r["predicted_letter"], r["correct_letter"], r["error"], r["raw_output"]]
            )

    print(f"==> Da ghi ket qua vao {out_dir}/summary.json va {out_dir}/per_question.csv")


if __name__ == "__main__":
    main()
