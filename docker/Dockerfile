# Dong goi giai phap cho vong 1 so loai (Viettel AI Race 2026).
#
# Bake san model weights vao image LUC BUILD (khong tai luc container chay)
# de tuan thu luat anti-cheat: "Khong goi network external tu inference
# server trong qua trinh serving" (Statement.txt). Base image la ban BTC
# dung lam baseline (vllm/vllm-openai:v0.22.1) de dam bao tuong thich voi
# entrypoint/CLI flags trong docker-compose.yml.
#
# Build (tu thu muc goc repo, vi context can bao gom ca thu muc docker/):
#   docker build -f docker/Dockerfile -t <dockerhub-user>/<repo>:latest .
#
# Trong thuc te build chay tren GitHub Actions (xem
# .github/workflows/docker-build-push.yml), khong can Docker cai o may local.

FROM vllm/vllm-openai:v0.22.1

# Model public tren HuggingFace, khong can token. Khi BTC cong bo hash commit
# co dinh, doi MODEL_REVISION cho khop (hien tai dung "main").
ARG MODEL_REPO=Qwen/Qwen3.5-2B
ARG MODEL_REVISION=main

RUN pip install --no-cache-dir -U "huggingface_hub[cli]" && \
    huggingface-cli download "${MODEL_REPO}" --revision "${MODEL_REVISION}" \
        --local-dir /model

# entrypoint/command that su duoc chi dinh boi docker-compose.yml luc BTC
# chay container (dung dinh dang co san, xem docker/docker-compose.yml) -
# khong dat CMD/ENTRYPOINT rieng o day de tranh xung dot.
