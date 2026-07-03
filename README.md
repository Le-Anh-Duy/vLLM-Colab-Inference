# vLLM Colab Setup

Dev repo cho track 3 — Viettel AI Race 2026 (LLM Inference Optimization Challenge). Xem đề bài đầy đủ ở [`Statement.txt`](Statement.txt).

Workflow: code/script được viết và version-control ở đây (local), sau đó push lên GitHub và pull xuống Colab để chạy — không cài/chạy thư viện nặng (torch, vllm, CUDA...) ở local.

## Bối cảnh

- Môi trường chấm điểm chính thức của BTC: NVIDIA H200, Ubuntu 22.04 LTS, CUDA 12.x, driver do BTC cung cấp sẵn.
- Colab (T4/L4/A100 tuỳ phiên) chỉ dùng để dev/thử nghiệm nhanh — cấu hình mặc định trong repo (CUDA 13.0 runtime, vllm 0.24.0 bản `+cu130`) là cấu hình đã chạy được trên Colab, **không nhất thiết trùng** với môi trường H200 thật.
- Model chính thức của cuộc thi sẽ do BTC chỉ định theo từng vòng (Dense Transformer, BF16, Apache 2.0, tải từ HuggingFace). Model mặc định để dev/benchmark hiện tại là [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.5B tham số, BF16, Apache 2.0) — đủ lớn để phản ánh tương đối chân thực latency/TPOT của một dense transformer thật, nhưng vẫn nhỏ gọn (~3GB weights) để chạy nhanh và không tràn VRAM trên GPU Colab (T4 15GB trở lên).
- vLLM publish sẵn wheel cho nhiều bản CUDA (vd `+cu128`, `+cu129`, `+cu130`) của cùng một version — đổi CUDA target chỉ là đổi biến `CUDA_VERSION`, không phải sửa code. Xem 3 kịch bản bên dưới.

### 3 kịch bản chạy `scripts/setup_vllm.sh`

| Kịch bản | Lệnh | Behavior |
|---|---|---|
| **A. Dev trên Colab** (mặc định) | `bash scripts/setup_vllm.sh` | Tự `apt install cuda-cudart-13-0` (Colab base image không có sẵn) + cài vllm/torch bản `+cu130`. |
| **B. Nộp bài cho BTC** | `SKIP_CUDA=1 CUDA_VERSION=12.8 bash scripts/setup_vllm.sh` | **Không đụng vào CUDA hệ thống** (dùng đúng CUDA/driver BTC đã cấp sẵn), chỉ pip cài vllm/torch bản `+cu128` khớp với CUDA 12.x của họ. |
| **C. Test trên Colab, giả lập giống lúc nộp** | `CUDA_VERSION=12.8 bash scripts/setup_vllm.sh` | Giống A nhưng cài `cuda-cudart-12-8` + vllm/torch bản `+cu128`, để kiểm tra trước khi nộp. |

**Về restart session:** nếu đây là lần setup đầu tiên trong một Colab session (kernel chưa từng `import torch`) thì không cần restart, dù chọn kịch bản nào. Nhưng nếu bạn đổi `CUDA_VERSION` *giữa chừng* trong cùng session (vd vừa chạy A xong giờ muốn thử C) thì **bắt buộc phải restart kernel** trước khi serve — mix lib CUDA cũ/mới trong cùng process Python sẽ lỗi. `notebooks/vllm_colab.ipynb` tự phát hiện việc này và tự restart kernel giúp bạn (xem cell 2), bạn chỉ cần chạy lại cell sau khi thấy kernel restart.

## Cấu trúc

```
scripts/
  setup_vllm.sh       # cài CUDA runtime + torch + vllm
  serve_vllm.sh       # khởi động vLLM OpenAI-compatible API server
  benchmark_traffic.py  # mo phong traffic + do TTFT/TPOT/ERS
notebooks/
  vllm_colab.ipynb  # clone repo, chạy setup + serve, test + benchmark
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
| `CUDA_VERSION` | `13.0` | Bản CUDA mục tiêu, dạng `MAJOR.MINOR` (vd `12.8`, `12.9`, `13.0`; patch như `12.8.1` cũng được, bị bỏ qua). Dạng nén `MAJORMINOR` (vd `128`) vẫn được chấp nhận. Suy ra apt package, lib path, torch index và wheel vllm tương ứng. |
| `CPU_ARCH` | `x86_64` | Kiến trúc CPU, dùng để dựng URL wheel vllm |
| `VLLM_VERSION` | `0.24.0` | Version vllm (dùng bản `vllm-${VLLM_VERSION}+cu${CUDA_VERSION}` chính thức từ GitHub Releases) |
| `CUDA_KEYRING_URL` | NVIDIA CUDA keyring (ubuntu2204) | URL gói `cuda-keyring` |
| `CUDA_PACKAGE` | suy ra từ `CUDA_VERSION`, vd `cuda-cudart-13-0` | Gói CUDA runtime cần cài |
| `CUDA_LIB_DIR` | suy ra từ `CUDA_VERSION`, vd `/usr/local/cuda-13.0/targets/x86_64-linux/lib` | Thư mục chứa `libcudart` |
| `CUDA_LIB_NAME` | suy ra từ `CUDA_VERSION`, vd `libcudart.so.13` | Tên file lib cần symlink |
| `TORCH_INDEX_URL` | suy ra từ `CUDA_VERSION`, vd `https://download.pytorch.org/whl/cu130` | Extra index pip cho torch |
| `VLLM_WHEEL_URL` | suy ra từ `VLLM_VERSION`+`CUDA_VERSION`+`CPU_ARCH` | URL wheel vllm chính thức, override nếu cần bản khác |
| `SKIP_CUDA` | `0` | `1` để bỏ qua bước `apt install` CUDA runtime — dùng khi CUDA/driver đã được môi trường (vd máy chấm BTC) cung cấp sẵn |
| `SKIP_PYTHON_DEPS` | `0` | `1` để bỏ qua bước cài torch/vllm |

### `scripts/serve_vllm.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Model để serve (HuggingFace repo id) |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Tỷ lệ VRAM vllm được phép dùng |
| `VLLM_EXTRA_ARGS` | *(rỗng)* | Chuỗi tham số bổ sung truyền thẳng cho `vllm.entrypoints.openai.api_server` (vd `--quantization fp8 --max-model-len 4096`) |

Script cũng nhận thêm tham số dòng lệnh trực tiếp, vd:

```bash
bash scripts/serve_vllm.sh --enable-prefix-caching
```

### `scripts/benchmark_traffic.py`

Mô phỏng traffic gửi tới server đang chạy (`serve_vllm.sh`) qua `/v1/chat/completions` (streaming), đo TTFT/TPOT từng request, và tính điểm ERS xấp xỉ theo đúng công thức ở mục 3.2 `Statement.txt` (interpolate Floor/Ceiling, trọng số `w`, hệ số luỹ thừa `gamma`).

```bash
pip install httpx   # 1 lần trên Colab, chưa có sẵn

# Trace giả lập (Poisson arrival), 2 req/s trong 30s:
python3 scripts/benchmark_traffic.py --synthetic --rps 2 --duration 30

# Trace có sẵn (JSONL, mỗi dòng: {"id","arrival_time","prompt","max_tokens"}):
python3 scripts/benchmark_traffic.py --trace-file trace.jsonl
```

Kết quả in ra console (TTFT/TPOT mean/p50/p95, ERS) và ghi vào `benchmark_results/summary.json` + `per_request.csv`.

**Quan trọng:** các ngưỡng `--floor-ttft`, `--ceil-ttft`, `--floor-tpot`, `--ceil-tpot`, `--w`, `--gamma` hiện là **placeholder** (chưa có số liệu chính thức từ BTC — xem mục 3.2 `Statement.txt`: "*Ngưỡng cụ thể của từng vòng sẽ được công bố*"). Khi BTC công bố, chỉ cần truyền lại qua CLI flag, không cần sửa code. ERS in ra chỉ mang tính tương đối để so sánh giữa các lần tối ưu, không phải điểm thi thật.
