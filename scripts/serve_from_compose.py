#!/usr/bin/env python3
"""Doc truc tiep docker/docker-compose.yml (entrypoint + command cua service
"model") va khoi dong vLLM voi DUNG NHUNG FLAG SE NOP CHO BTC - khong tu tay
copy flag qua serve_vllm.sh nua (tung bi lech: --swap-space/--disable-log-
requests da doi/bo trong compose ma quen sua serve_vllm.sh, gay loi that khi
BTC chay). Day la nguon su that duy nhat (single source of truth).

Chi thay the DUY NHAT gia tri cua "--model=/model" bang --model-override, vi
"/model" chi ton tai ben trong container that cua BTC (ho tu mount weight
vao do) - tren Colab khong co thu muc do, phai dung HF repo id thay the. Moi
flag khac giu nguyen 100% tu docker-compose.yml.

--override: dung khi GPU tren Colab KHONG ho tro mot flag nao do ve mat phan
cung (vd --kv-cache-dtype=fp8 can SM89+/Hopper, GPU Colab dang cap co the
chi la T4/SM75) - override CHI anh huong phien chay hien tai tren Colab,
KHONG sua docker-compose.yml (file se nop cho BTC van giu nguyen). Luon in
ro flag nao bi override de khong bi nham la da test dung cau hinh that.

Vi du:
    python3 scripts/serve_from_compose.py --model-override Qwen/Qwen3.5-2B
    python3 scripts/serve_from_compose.py --model-override Qwen/Qwen3.5-2B \
        --override kv-cache-dtype=auto
"""
from __future__ import annotations

import argparse
import subprocess
import sys

try:
    import yaml
except ImportError:
    print("Thieu thu vien pyyaml. Chay: pip install pyyaml", file=sys.stderr)
    raise


def load_argv(compose_path: str, service: str, model_override: str) -> list[str]:
    with open(compose_path, encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    try:
        svc = compose["services"][service]
    except KeyError:
        raise SystemExit(
            f"Khong tim thay service '{service}' trong {compose_path}. "
            f"Cac service co san: {list(compose.get('services', {}).keys())}"
        )

    argv = list(svc["entrypoint"]) + list(svc["command"])

    replaced = False
    for i, tok in enumerate(argv):
        if tok == "--model=/model":
            argv[i] = f"--model={model_override}"
            replaced = True
            break
    if not replaced:
        print(
            f"CANH BAO: khong tim thay '--model=/model' trong command cua "
            f"{compose_path} - co the file da doi cau truc, kiem tra lai.",
            file=sys.stderr,
        )

    return argv


def apply_overrides(argv: list[str], overrides: list[str]) -> list[str]:
    """Moi override dang 'flag=value'. Thay the token '--flag=<cu>' bang
    '--flag=<moi>' neu tim thay, neu khong thi them vao cuoi. Chi dung de
    testing tren Colab khi phan cung khac voi may cham that - KHONG dung de
    "chinh sua vinh vien" cau hinh (do phai sua truc tiep docker-compose.yml)."""
    result = list(argv)
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"--override phai co dang 'flag=value', nhan duoc: {ov!r}")
        flag, value = ov.split("=", 1)
        prefix = f"--{flag}="
        new_tok = f"{prefix}{value}"
        found = False
        for i, tok in enumerate(result):
            if tok.startswith(prefix):
                print(f"==> OVERRIDE (chi cho lan chay nay tren Colab, KHONG sua docker-compose.yml): {result[i]} -> {new_tok}")
                result[i] = new_tok
                found = True
                break
        if not found:
            print(f"==> OVERRIDE: them moi (khong co san trong compose): {new_tok}")
            result.append(new_tok)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compose-file", default="docker/docker-compose.yml")
    p.add_argument("--service", default="model")
    p.add_argument(
        "--model-override",
        default="Qwen/Qwen3.5-2B",
        help="HF repo id dung thay cho '/model' khi chay tren Colab (khong co trong file that nop cho BTC)",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help="Ghi de 1 flag CHI cho lan chay nay (vd khi GPU Colab khong ho tro fp8: --override kv-cache-dtype=auto). "
        "Co the lap lai nhieu lan. KHONG anh huong docker-compose.yml.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    argv = load_argv(args.compose_file, args.service, args.model_override)
    argv = apply_overrides(argv, args.override)

    print(f"==> Doc tu {args.compose_file} (service '{args.service}'), da thay --model=/model -> --model={args.model_override}")
    if args.override:
        print("==> CHU Y: dang chay voi override khac so voi docker-compose.yml (xem tren) - file se nop cho BTC KHONG doi.")
    print("==> Lenh se chay:")
    print("    " + " ".join(argv))
    print("=" * 60)

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
