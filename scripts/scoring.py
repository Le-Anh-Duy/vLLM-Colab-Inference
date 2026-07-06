"""Cong thuc cham diem dung chung, lay tu Statement.txt (vong 1 so loai, track
3 Viettel AI Race 2026) - dung boi benchmark_traffic.py (ERS) va eval_gpqa.py
(accuracy gate).

QUAN TRONG: accuracy va baseline_accuracy dung THANG TY LE 0..1 (vd 0.4 = 40%),
KHONG PHAI 0..100 - dung theo dinh nghia trong Statement.txt muc 2 (baseline
mac dinh dat 0.4)."""
from __future__ import annotations


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def accuracy_decay(delta: float) -> float:
    """f(delta) - Statement.txt muc 2 (Accuracy Gate). delta = baseline_accuracy
    - accuracy cua doi, ca hai theo thang 0..1 (vd delta=0.1 nghia la lech 10
    diem % tuyet doi)."""
    if delta <= 0.10:
        return 1.0
    if delta < 0.16:
        return 1.0 - (delta - 0.10) / 0.06
    return 0.0


def final_score(ers: float, baseline_accuracy: float, team_accuracy: float) -> dict:
    """Score = 100 * ERS * f(delta) - Statement.txt muc 2."""
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
