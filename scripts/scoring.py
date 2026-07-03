"""Cong thuc cham diem dung chung, lay tu Statement.txt (track 3, Viettel AI
Race 2026) - dung boi benchmark_traffic.py (ERS) va eval_gpqa.py (accuracy
gate)."""
from __future__ import annotations


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def accuracy_decay(delta: float) -> float:
    """f(delta) - Statement.txt muc 3.3. delta = baseline_accuracy - accuracy
    cua doi (tinh theo % diem, vd 72.0 nghia la 72%)."""
    if delta <= 10:
        return 1.0
    if delta < 16:
        return 1.0 - (delta - 10) / 6
    return 0.0


def final_score(ers: float, baseline_accuracy: float, team_accuracy: float) -> dict:
    """Score = 100 * ERS * f(delta) - Statement.txt muc 3.4."""
    delta = baseline_accuracy - team_accuracy
    penalty = accuracy_decay(delta)
    return {
        "ers": ers,
        "baseline_accuracy": baseline_accuracy,
        "team_accuracy": team_accuracy,
        "delta": delta,
        "accuracy_penalty_f_delta": penalty,
        "score": 100 * ers * penalty,
    }
