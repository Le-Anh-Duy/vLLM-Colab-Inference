# vLLM Colab Setup

Dev repo cho track 3 — Viettel AI Race 2026 (LLM Inference Optimization Challenge). Xem đề bài đầy đủ ở [`Statement.txt`](Statement.txt).

Workflow: code/script được viết và version-control ở đây (local), sau đó push lên GitHub và pull xuống Colab để chạy — không cài/chạy thư viện nặng (torch, vllm, CUDA...) ở local.

## Bối cảnh

- Môi trường chấm điểm chính thức của BTC: NVIDIA H200, Ubuntu 22.04 LTS, CUDA 12.x.
- Colab (T4/L4/A100 tuỳ phiên) chỉ dùng để dev/thử nghiệm nhanh — cấu hình mặc định trong repo (CUDA 13.0 runtime, torch 2.11.0, vllm 0.24.0) là cấu hình đã chạy được trên Colab, **không nhất thiết trùng** với môi trường H200 thật. Khi cần khớp CUDA 12.x, override qua biến môi trường (xem bên dưới).
- Model chính thức của cuộc thi sẽ do BTC chỉ định theo từng vòng (Dense Transformer, BF16, Apache 2.0, tải từ HuggingFace). `facebook/opt-125m` hiện chỉ là model nhỏ để test pipeline.

## Cấu trúc

```
scripts/
  setup_vllm.sh   # cài CUDA runtime + torch + vllm
  serve_vllm.sh   # khởi động vLLM OpenAI-compatible API server
notebooks/
  vllm_colab.ipynb  # clone repo, chạy setup + serve, test bằng OpenAI client
```

## Sử dụng trên Colab

1. Push repo lên GitHub.
2. Mở `notebooks/vllm_colab.ipynb` trên Colab (Runtime → chọn GPU).
3. Sửa `REPO_URL` ở cell đầu thành URL repo GitHub của bạn.
4. Chạy tuần tự các cell: clone → setup → serve (background) → test request.

## Override cấu hình

Tất cả tham số đều có giá trị mặc định, override bằng cách set biến môi trường trước khi gọi script (trong Colab dùng `%env TEN_BIEN=gia_tri` hoặc `os.environ["TEN_BIEN"] = "..."`).

### `scripts/setup_vllm.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CUDA_KEYRING_URL` | NVIDIA CUDA 13 keyring (ubuntu2204) | URL gói `cuda-keyring` |
| `CUDA_PACKAGE` | `cuda-cudart-13-0` | Gói CUDA runtime cần cài |
| `CUDA_LIB_DIR` | `/usr/local/cuda-13.0/targets/x86_64-linux/lib` | Thư mục chứa `libcudart` |
| `CUDA_LIB_NAME` | `libcudart.so.13` | Tên file lib cần symlink |
| `TORCH_VERSION` | `2.11.0` | Version torch |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu130` | Index pip cho torch (đổi sang `cu121`/`cu124`... nếu cần khớp CUDA 12.x) |
| `VLLM_VERSION` | `0.24.0` | Version vllm |
| `SKIP_CUDA` | `0` | `1` để bỏ qua bước cài CUDA (vd đã cài rồi) |
| `SKIP_PYTHON_DEPS` | `0` | `1` để bỏ qua bước cài torch/vllm |

### `scripts/serve_vllm.sh`

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_NAME` | `facebook/opt-125m` | Model để serve (HuggingFace repo id) |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Tỷ lệ VRAM vllm được phép dùng |
| `VLLM_EXTRA_ARGS` | *(rỗng)* | Chuỗi tham số bổ sung truyền thẳng cho `vllm.entrypoints.openai.api_server` (vd `--quantization fp8 --max-model-len 4096`) |

Script cũng nhận thêm tham số dòng lệnh trực tiếp, vd:

```bash
bash scripts/serve_vllm.sh --enable-prefix-caching
```
