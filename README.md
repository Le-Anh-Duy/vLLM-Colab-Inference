# vLLM Colab Setup

Dev repo cho track 3 — Viettel AI Race 2026 (LLM Inference Optimization Challenge). Xem đề bài đầy đủ ở [`Statement.txt`](Statement.txt).

Workflow: code/script được viết và version-control ở đây (local), sau đó push lên GitHub và pull xuống Colab để chạy — không cài/chạy thư viện nặng (torch, vllm, CUDA...) ở local.

## Bối cảnh

- Môi trường chấm điểm chính thức của BTC: NVIDIA H200, Ubuntu 22.04 LTS, CUDA 12.x, driver do BTC cung cấp sẵn.
- Colab (T4/L4/A100 tuỳ phiên) chỉ dùng để dev/thử nghiệm nhanh — cấu hình mặc định trong repo (CUDA 13.0 runtime, torch 2.11.0, vllm 0.24.0) là cấu hình đã chạy được trên Colab, **không nhất thiết trùng** với môi trường H200 thật.
- Model chính thức của cuộc thi sẽ do BTC chỉ định theo từng vòng (Dense Transformer, BF16, Apache 2.0, tải từ HuggingFace). Model mặc định để dev/benchmark hiện tại là [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (1.5B tham số, BF16, Apache 2.0) — đủ lớn để phản ánh tương đối chân thực latency/TPOT của một dense transformer thật, nhưng vẫn nhỏ gọn (~3GB weights) để chạy nhanh và không tràn VRAM trên GPU Colab (T4 15GB trở lên).
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
  benchmark_traffic.py  # mô phỏng traffic + đo TTFT/TPOT/ERS
  eval_gpqa.py          # chấm accuracy GPQA Diamond (Accuracy Gate)
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
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Model để serve (HuggingFace repo id) |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Tỷ lệ VRAM vllm được phép dùng (an toàn cho GPU dành riêng như Colab; xuống thấp hơn nếu chia sẻ GPU với process khác) |
| `ENABLE_PREFIX_CACHING` | `1` | `0` để tắt (`--no-enable-prefix-caching`) — dùng khi cần A/B so sánh có/không prefix caching. vLLM bản hiện tại đã bật mặc định, cờ này chỉ để chủ động kiểm soát/so sánh. |
| `KV_CACHE_DTYPE` | *(rỗng = `auto`)* | Set `fp8` để quantize KV cache (Statement.txt mục 5: KV Cache Optimization) — tăng concurrency/context nhưng có thể ảnh hưởng accuracy, **luôn chấm lại bằng `scripts/eval_gpqa.py` sau khi bật**. |
| `VLLM_EXTRA_ARGS` | *(rỗng)* | Chuỗi tham số bổ sung truyền thẳng cho `vllm.entrypoints.openai.api_server` (vd `--quantization fp8 --max-model-len 4096`) |

Script cũng nhận thêm tham số dòng lệnh trực tiếp, vd:

```bash
bash scripts/serve_vllm.sh --max-model-len 4096
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

### `scripts/eval_gpqa.py`

Chấm accuracy trên GPQA Diamond cho server đang chạy, phục vụ **Accuracy Gate** (mục 3.3 `Statement.txt`) — bắt buộc phải chạy lại script này sau MỖI lần đổi quantization/tối ưu có khả năng ảnh hưởng chất lượng đầu ra (KV cache FP8, weight quantization, speculative decoding...), để biết `Δ` (accuracy drop so với baseline) có vượt ngưỡng bị phạt hay không.

```bash
pip install httpx

# BTC chua cong bo 100 cau hoi co dinh - dung file JSONL cua rieng ban truoc:
# {"id": "0", "question": "...", "choices": ["A..","B..","C..","D.."], "answer_index": 2}
python3 scripts/eval_gpqa.py --questions-file gpqa_diamond.jsonl --baseline-accuracy 70.0

# Hoac tu test bang bo GPQA Diamond public tren HuggingFace (dataset gated,
# can `pip install datasets` + chap nhan dieu khoan/HF_TOKEN - KHONG phai bo
# cau hoi BTC dung de cham that):
python3 scripts/eval_gpqa.py --hf-fallback --num-questions 100
```

Kết quả in accuracy (%) ra console, và nếu truyền `--baseline-accuracy` sẽ tính luôn `Δ` và hệ số phạt `f(Δ)` theo đúng công thức piecewise trong `Statement.txt`. Ghi chi tiết vào `gpqa_results/summary.json` + `per_question.csv` (kèm câu trả lời thô của model để soát lại khi cần).

**Khi BTC công bố file 100 câu hỏi chính thức:** chỉ cần trỏ `--questions-file` vào file đó, không sửa code. `scripts/scoring.py` chứa chung công thức `f(Δ)` và `clamp01` dùng bởi cả `benchmark_traffic.py` và `eval_gpqa.py`.
