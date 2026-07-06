# vLLM Colab Setup

Dev repo cho track 3 — Viettel AI Race 2026 (LLM Inference Optimization Challenge). Xem đề bài đầy đủ ở [`Statement.txt`](Statement.txt).

Workflow: code/script được viết và version-control ở đây (local), sau đó push lên GitHub và pull xuống Colab để chạy — không cài/chạy thư viện nặng (torch, vllm, CUDA...) ở local.

## Bối cảnh

- **Vòng 1 sơ loại** (02/07 - 30/07/2026): serve `Qwen/Qwen3.5-2B` xử lý file trace 120 request (`baseline-and-input/trace-round1.jsonl`), tối đa hoá **ERS** trong khi vẫn qua **Accuracy Gate** (GPQA Diamond). Xem đầy đủ ở [`Statement.txt`](Statement.txt).
- **Hạ tầng chấm điểm thật:** mỗi lượt chấm chỉ được cấp **1 lát MiG H200: 18GB VRAM, 3 CPU, 8GB RAM** — không phải full H200. Nộp bài bằng cách push Docker image public lên Docker Hub + nộp `docker-compose.yml` cho BTC pull về chạy tự động (xem mục "Nộp bài bằng Docker" bên dưới). File compose mẫu của BTC: `baseline-and-input/docker-compose-baseline.yml`.
- **Ngưỡng điểm vòng 1 (đã là số thật, không còn placeholder):** `F_ttft=100ms, C_ttft=1500ms, F_tpot=20ms, C_tpot=45ms, γ=2, w=0.5`; Accuracy Gate: `baseline_accuracy=0.4` (thang 0..1), phạt bắt đầu từ `Δ>0.10`, về 0 điểm accuracy ở `Δ≥0.16`.
- **Hệ quả quan trọng:** `--max-model-len=262144` (262K context) trên chỉ 18GB VRAM là rất căng — KV cache quantization (FP8/INT8) gần như bắt buộc chứ không còn là tối ưu tuỳ chọn, nếu không sẽ không đủ chỗ cho KV cache ở context dài với bất kỳ concurrency nào.
- Colab (T4/L4/A100 tuỳ phiên) chỉ dùng để dev/thử nghiệm nhanh — VRAM thường **nhiều hơn** 18GB của MiG slice thật, nên test "chạy được trên Colab" không đảm bảo chạy được trên máy chấm; khi cần mô phỏng sát VRAM 18GB, giảm `MAX_MODEL_LEN` hoặc bật `KV_CACHE_DTYPE=fp8`.
- vLLM có publish wheel cho một vài bản CUDA cụ thể qua GitHub Releases (vd `+cu129`), nhưng **mỗi version chỉ có một số biến thể nhất định** — không phải version nào cũng có `+cu128`/`+cu130`. Luôn kiểm tra asset thật tại `https://github.com/vllm-project/vllm/releases/tag/v<VLLM_VERSION>` trước khi đổi `CUDA_VERSION` cho kịch bản B/C.

### 3 kịch bản chạy `scripts/setup_vllm.sh`

| Kịch bản | Lệnh | Behavior |
|---|---|---|
| **A. Dev trên Colab** (mặc định) | `bash scripts/setup_vllm.sh` | Tự `apt install cuda-cudart-13-0` (Colab base image không có sẵn) + `pip install torch==2.11.0` (extra-index cu130) + `pip install vllm==0.24.0` (bản mặc định, không suffix CUDA) — đúng công thức đã chạy ổn định từ đầu. |
| **B. Nộp bài cho BTC** | `SKIP_CUDA=1 PINNED_CUDA_WHEEL=1 CUDA_VERSION=12.9 bash scripts/setup_vllm.sh` | **Không đụng vào CUDA hệ thống** (dùng đúng CUDA/driver BTC đã cấp sẵn), pip cài đúng wheel `vllm+cu129` từ GitHub Release khớp CUDA 12.x của họ (có kiểm tra URL tồn tại trước khi cài). |
| **C. Test trên Colab, giả lập giống lúc nộp** | `PINNED_CUDA_WHEEL=1 CUDA_VERSION=12.9 bash scripts/setup_vllm.sh` | Giống B nhưng vẫn tự `apt install cuda-cudart-12-9` vì Colab không có sẵn CUDA runtime như máy BTC. |

**Quan trọng — 2 biến CUDA tách biệt, không dùng lẫn:** `CUDA_VERSION` chỉ có tác dụng khi `PINNED_CUDA_WHEEL=1` (kịch bản B/C). Ở kịch bản A (`PINNED_CUDA_WHEEL=0`), script dùng `PLAIN_CUDA_VERSION` (mặc định `13.0`, **cố định** vì bản `vllm==VERSION` mặc định luôn được biên dịch với CUDA 13 runtime — set `CUDA_VERSION` sẽ *không* ảnh hưởng gì tới kịch bản A. Set nhầm `CUDA_VERSION` trong khi vẫn ở `PINNED_CUDA_WHEEL=0` từng gây lỗi `libcudart.so.13: cannot open shared object file` do cudart bị cài sai bản.

**Về restart session:** nếu đây là lần setup đầu tiên trong một Colab session (kernel chưa từng `import torch`) thì không cần restart, dù chọn kịch bản nào. Nhưng nếu bạn đổi cấu hình CUDA *giữa chừng* trong cùng session (vd vừa chạy A xong giờ muốn thử C) thì **bắt buộc phải restart kernel** trước khi serve — mix lib CUDA cũ/mới trong cùng process Python sẽ lỗi. `notebooks/vllm_colab.ipynb` tự phát hiện việc này và tự restart kernel giúp bạn (xem cell 2), bạn chỉ cần chạy lại cell sau khi thấy kernel restart.

## Cấu trúc

```
scripts/
  setup_vllm.sh         # cài CUDA runtime + torch + vllm
  serve_vllm.sh         # khởi động vLLM OpenAI-compatible API server
  scoring.py            # công thức chấm điểm dùng chung (ERS clamp, accuracy decay f(Δ))
  benchmark_traffic.py  # đo TTFT/TPOT/ERS trên trace thật hoặc trace giả lập
  eval_gpqa.py          # chấm accuracy GPQA Diamond (Accuracy Gate)
notebooks/
  vllm_colab.ipynb  # clone repo, chạy setup + serve, test + benchmark
baseline-and-input/
  trace-round1.jsonl            # trace 120 request thật của BTC (vòng 1)
  docker-compose-baseline.yml   # compose mẫu của BTC (tham khảo, không sửa)
docker/
  Dockerfile          # bake weights Qwen3.5-2B vào image lúc build
  docker-compose.yml  # file nộp cho BTC, trỏ vào image trên Docker Hub
.github/workflows/
  docker-build-push.yml  # build + push image lên Docker Hub qua GitHub Actions
```

## Sử dụng trên Colab

1. Push repo lên GitHub.
2. Mở `notebooks/vllm_colab.ipynb` trên Colab (Runtime → chọn GPU).
3. Sửa `REPO_URL` ở cell 1 thành URL repo GitHub của bạn.
4. Cell 1b: chọn 1 trong 3 kịch bản CUDA ở trên (mặc định là A, không cần sửa gì).
5. Chạy tuần tự các cell: clone → setup → serve (background) → test request. Nếu cell 2 tự restart kernel (do đổi `CUDA_VERSION` giữa chừng), chạy lại từ cell 1b sau khi kernel restart xong.

## Override cấu hình

Tất cả tham số đều có giá trị mặc định, override bằng cách set biến môi trường trước khi gọi script (trong Colab dùng `%env TEN_BIEN=gia_tri` hoặc `os.environ["TEN_BIEN"] = "..."`).

### `scripts/setup_vllm.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `PINNED_CUDA_WHEEL` | `0` | `0` = cài kiểu thường (`pip install torch==TORCH_VERSION` qua extra-index rồi `pip install vllm==VLLM_VERSION` bản mặc định — dùng cho dev Colab, CUDA runtime cố định theo `PLAIN_CUDA_VERSION`). `1` = tải đúng wheel `vllm+cuXXX` từ GitHub Release ứng với `CUDA_VERSION` — dùng khi cần khớp chính xác CUDA của môi trường đích (vd BTC), có kiểm tra URL tồn tại trước khi cài. |
| `PLAIN_CUDA_VERSION` | `13.0` | **Chỉ dùng khi `PINNED_CUDA_WHEEL=0`.** CUDA runtime mà bản `vllm==VLLM_VERSION` mặc định (không suffix) yêu cầu — cố định theo build của vllm, không tự đổi tuỳ ý được trừ khi đổi `VLLM_VERSION` sang bản khác. |
| `CUDA_VERSION` | `12.9` | **Chỉ dùng khi `PINNED_CUDA_WHEEL=1`.** Bản CUDA mục tiêu, dạng `MAJOR.MINOR` (vd `12.9`; patch như `12.9.1` cũng được, bị bỏ qua). Dạng nén `MAJORMINOR` (vd `129`) vẫn được chấp nhận. Suy ra apt package, lib path, torch index và wheel vllm tương ứng. |
| `TORCH_VERSION` | `2.11.0` | Version torch, chỉ dùng khi `PINNED_CUDA_WHEEL=0` |
| `CPU_ARCH` | `x86_64` | Kiến trúc CPU, dùng để dựng URL wheel vllm khi `PINNED_CUDA_WHEEL=1` |
| `VLLM_VERSION` | `0.24.0` | Version vllm |
| `VLLM_MANYLINUX_TAG` | `manylinux_2_28` | Tag manylinux trong tên wheel GitHub Release, chỉ dùng khi `PINNED_CUDA_WHEEL=1` — kiểm tra đúng tag tại trang release nếu 404 |
| `CUDA_KEYRING_URL` | NVIDIA CUDA keyring (ubuntu2204) | URL gói `cuda-keyring` |
| `CUDA_PACKAGE` | suy ra từ `CUDA_VERSION`, vd `cuda-cudart-13-0` | Gói CUDA runtime cần cài |
| `CUDA_LIB_DIR` | suy ra từ `CUDA_VERSION`, vd `/usr/local/cuda-13.0/targets/x86_64-linux/lib` | Thư mục chứa `libcudart` |
| `CUDA_LIB_NAME` | suy ra từ `CUDA_VERSION`, vd `libcudart.so.13` | Tên file lib cần symlink |
| `TORCH_INDEX_URL` | suy ra từ `CUDA_VERSION`, vd `https://download.pytorch.org/whl/cu130` | Extra index pip cho torch |
| `VLLM_WHEEL_URL` | suy ra từ `VLLM_VERSION`+`CUDA_VERSION`+`CPU_ARCH`+`VLLM_MANYLINUX_TAG` | URL wheel vllm khi `PINNED_CUDA_WHEEL=1`, override nếu cần bản khác |
| `SKIP_CUDA` | `0` | `1` để bỏ qua bước `apt install` CUDA runtime — dùng khi CUDA/driver đã được môi trường (vd máy chấm BTC) cung cấp sẵn |
| `SKIP_PYTHON_DEPS` | `0` | `1` để bỏ qua bước cài torch/vllm |

### `scripts/serve_vllm.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3.5-2B` | Model để load (HuggingFace repo id, dùng cho `--model`) |
| `SERVED_MODEL_NAME` | `Qwen3.5-2B` | Tên model client gọi qua API (`--served-model-name`) — phải khớp field `"model"` trong request/trace, KHÔNG phải HF repo id |
| `MAX_MODEL_LEN` | `262144` | Context length tối đa (khớp baseline compose). Rất tốn KV cache trên GPU nhỏ — giảm giá trị này khi test trên Colab nếu OOM |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Tỷ lệ VRAM vllm được phép dùng. Trên Colab (GPU dành riêng, thường >18GB) để 0.85 là an toàn; khi mô phỏng đúng MiG 18GB có thể cần theo sát baseline (`0.95`) |
| `TENSOR_PARALLEL_SIZE` | `1` | Khớp baseline compose (1 GPU/MiG instance, không có multi-GPU) |
| `ENABLE_PREFIX_CACHING` | `1` | `0` để tắt (`--no-enable-prefix-caching`) — dùng khi cần A/B so sánh có/không prefix caching. |
| `KV_CACHE_DTYPE` | *(rỗng = `auto`)* | Set `fp8` để quantize KV cache (Statement.txt mục 3: KV Cache & Memory) — gần như bắt buộc với context 262K trên 18GB VRAM, nhưng có thể ảnh hưởng accuracy, **luôn chấm lại bằng `scripts/eval_gpqa.py` sau khi bật**. |
| `VLLM_EXTRA_ARGS` | *(rỗng)* | Chuỗi tham số bổ sung truyền thẳng cho `vllm.entrypoints.openai.api_server` (vd `--quantization fp8`) |

Script cũng nhận thêm tham số dòng lệnh trực tiếp, vd:

```bash
bash scripts/serve_vllm.sh --max-model-len 8192
```

### `scripts/benchmark_traffic.py`

Gửi traffic tới server đang chạy (`serve_vllm.sh`) qua `/v1/chat/completions` (streaming), đo TTFT/TPOT từng request, và tính **ERS** đúng công thức thật ở mục 2 `Statement.txt` (Floor/Ceiling/`w`/`gamma` mặc định = số thật vòng 1 sơ loại).

```bash
pip install httpx   # 1 lần trên Colab, chưa có sẵn

# Trace THAT cua BTC (120 request, dung file nay de uoc luong ERS sat nhat):
python3 scripts/benchmark_traffic.py --trace-file baseline-and-input/trace-round1.jsonl

# Trace giả lập (Poisson arrival) khi muon test nhanh khong can trace that:
python3 scripts/benchmark_traffic.py --synthetic --rps 2 --duration 30
```

`--trace-file` tự nhận diện 2 định dạng: JSONL thật của BTC (`{"request_id", "timestamp_ms", "body": {...}}`, dùng luôn cho `trace-round1.jsonl`) hoặc JSONL generic tự viết (`{"id", "arrival_time", "prompt", "max_tokens"}`).

Kết quả in ra console (TTFT/TPOT mean/p50/p95, ERS) và ghi vào `benchmark_results/summary.json` + `per_request.csv`. Nhớ `--model` phải khớp `SERVED_MODEL_NAME` (`Qwen3.5-2B`), không phải HF repo id.

**Lưu ý:** ngưỡng Floor/Ceiling/`w`/`gamma` là số thật của **vòng 1 sơ loại** — các vòng sau BTC có thể công bố số khác, lúc đó chỉ cần truyền lại qua CLI flag (`--floor-ttft`, `--ceil-ttft`, ...), không cần sửa code.

### `scripts/eval_gpqa.py`

Chấm accuracy trên GPQA Diamond cho server đang chạy, phục vụ **Accuracy Gate** (mục 2 `Statement.txt`) — bắt buộc phải chạy lại script này sau MỖI lần đổi quantization/tối ưu có khả năng ảnh hưởng chất lượng đầu ra (KV cache FP8, weight quantization, speculative decoding...), để biết `Δ` (accuracy drop so với baseline) có vượt ngưỡng bị phạt hay không.

```bash
pip install httpx

# BTC chua cong bo 100 cau hoi co dinh - dung file JSONL cua rieng ban truoc:
# {"id": "0", "question": "...", "choices": ["A..","B..","C..","D.."], "answer_index": 2}
python3 scripts/eval_gpqa.py --questions-file gpqa_diamond.jsonl

# Hoac tu test bang bo GPQA Diamond public tren HuggingFace (dataset gated,
# can `pip install datasets` + chap nhan dieu khoan/HF_TOKEN - KHONG phai bo
# cau hoi BTC dung de cham that):
python3 scripts/eval_gpqa.py --hf-fallback --num-questions 100
```

**Lưu ý đơn vị:** accuracy và `--baseline-accuracy` dùng thang **0..1** (khớp Statement.txt: `baseline_accuracy` mặc định `0.4` = 40%), không phải phần trăm 0..100. Mặc định `--baseline-accuracy` đã là `0.4`. Kết quả in accuracy + `Δ` + hệ số phạt `f(Δ)` ra console, ghi chi tiết vào `gpqa_results/summary.json` + `per_question.csv` (kèm câu trả lời thô của model để soát lại khi cần).

**Khi BTC công bố file 100 câu hỏi chính thức:** chỉ cần trỏ `--questions-file` vào file đó, không sửa code. `scripts/scoring.py` chứa chung công thức `f(Δ)` và `clamp01` dùng bởi cả `benchmark_traffic.py` và `eval_gpqa.py`.

## Nộp bài bằng Docker

Quy trình nộp bài của BTC: đóng gói giải pháp thành Docker image → push lên Docker Hub public → nộp `docker-compose.yml` trỏ vào image đó qua Portal → BTC tự pull về chạy trên MiG H200 và benchmark.

```
docker/
  Dockerfile          # FROM vllm/vllm-openai:v0.22.1, bake weights Qwen3.5-2B luc build
  docker-compose.yml  # nop cho BTC - mirror docker-compose-baseline.yml, tro ve image cua minh
.github/workflows/
  docker-build-push.yml  # build + push image len Docker Hub, chay tren GitHub Actions
```

**Không cần cài Docker cục bộ** — build chạy trên GitHub Actions (runner Ubuntu có sẵn Docker daemon), bake sẵn weights model vào image lúc build (tránh gọi mạng ngoài lúc serving, đúng luật anti-cheat "không pull model dynamically"), rồi push thẳng lên Docker Hub bằng access token.

### Setup 1 lần: thêm Docker Hub secrets vào GitHub repo

1. Đăng nhập [Docker Hub](https://hub.docker.com) (user `duylemeow`) → **Account Settings → Security → New Access Token** → đặt tên (vd `github-actions`), quyền `Read & Write` → copy token (chỉ hiện 1 lần).
2. Vào repo trên GitHub → **Settings → Secrets and variables → Actions → New repository secret**, tạo 2 secret:
   - `DOCKERHUB_USERNAME` = `duylemeow`
   - `DOCKERHUB_TOKEN` = token vừa tạo ở bước 1
3. Xong — không cần làm lại trừ khi token hết hạn/bị thu hồi.

### Chạy build

Vào tab **Actions** trên GitHub → chọn workflow **"Build and push Docker image"** → **Run workflow** (nhánh chứa `docker/`). Workflow cũng tự chạy khi có thay đổi trong `docker/**` được push lên `main`.

Build sẽ mất một lúc (tải base image ~10GB+ + tải weights Qwen3.5-2B) — theo dõi log trực tiếp trong tab Actions. Xong, image xuất hiện tại `hub.docker.com/r/duylemeow/vllm-qwen35-2b`.

**Giới hạn quan trọng:** GitHub Actions free runner **không có GPU**, nên CI chỉ smoke-test được `import vllm` + kiểm tra `/model` có file (bake weight thành công), **không** chạy được benchmark suy luận thật. Hành vi/hiệu năng/optimization vẫn phải xác nhận riêng qua Colab (CLI flags giống hệt `docker/docker-compose.yml`) trước khi build — Docker chỉ là lớp đóng gói cuối cùng.

### Trước khi nộp thật

1. Đảm bảo mọi flag tối ưu (kv-cache-dtype, quantization...) đã test qua `scripts/benchmark_traffic.py` + `scripts/eval_gpqa.py` trên Colab và pass Accuracy Gate.
2. Copy đúng các flag đó vào `command:` trong `docker/docker-compose.yml` (giữ nguyên các dòng có comment "Don't change this to vllm-server").
3. Chạy workflow build + push.
4. Nộp `docker/docker-compose.yml` qua Portal của BTC.
