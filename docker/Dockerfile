# Dong goi giai phap cho vong 1 so loai (Viettel AI Race 2026).
#
# LUU Y: da xac nhan bang thuc nghiem la KHONG BAT BUOC dung Dockerfile nay -
# nop thang docker-compose.yml voi image cong khai vllm/vllm-openai (khong
# bake weight) van chay va ra diem, vi ha tang BTC tu cap /model cho container
# (xem comment trong docker/docker-compose.yml). Dockerfile nay chi can thiet
# khi giai phap co code/kernel rieng KHONG co san trong image goc (vd patch
# vllm, custom kernel, thu vien them...). Neu optimization chi la CLI flags,
# dung thang docker-compose.yml tro vao image cong khai la du, khong can build
# gi ca.
#
# Neu van muon/can bake weights vao image LUC BUILD (khong tai luc container
# chay) de tuan thu luat anti-cheat: "Khong goi network external tu inference
# server trong qua trinh serving" (Statement.txt), dung Dockerfile nay.
#
# BTC dung vllm/vllm-openai:v0.22.1 lam baseline THAM KHAO, nhung Statement.txt
# muc 3 cho phep tu do chon framework/version. Da tung chay on dinh voi vllm
# 0.24.0 tren Colab (cung entrypoint "python3 -m
# vllm.entrypoints.openai.api_server" ma BTC yeu cau), va vllm/vllm-openai:
# v0.24.0 co san tren Docker Hub - nen dung ban nay de co cac cai tien/fix moi
# hon 0.22.1.
#
# Build (tu thu muc goc repo, vi context can bao gom ca thu muc docker/):
#   docker build -f docker/Dockerfile -t <dockerhub-user>/<repo>:latest .
#
# Trong thuc te build chay tren GitHub Actions (xem
# .github/workflows/docker-build-push.yml), khong can Docker cai o may local.

FROM vllm/vllm-openai:v0.24.0

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
