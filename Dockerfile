ARG BASE_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:b47c19d66633f27cbe37da13fbc580363f466c08b9529feab1eecb1a4b904bf1
FROM ${BASE_IMAGE}
ARG INSTALL_DEPS=0
# Keep only two filesystem layers: the retained local image already has 117.
COPY . /opt/recipe/
RUN if [ "$INSTALL_DEPS" = 1 ]; then python3 -m pip install --no-cache-dir -r /opt/recipe/requirements-runtime.txt; fi \
 && cp -a /opt/recipe/vendor/tensorfold/. /usr/local/lib/python3.12/dist-packages/tensorfold/ \
 && cp -a /opt/recipe/vendor/b12x/. /usr/local/lib/python3.12/dist-packages/b12x/ \
 && python3 /opt/recipe/scripts/check_runtime.py
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TENSORFOLD_NO_UPDATE_CHECK=1
LABEL org.opencontainers.image.title="GLM-5.3-Flash-EXLR8-K4-TensorFold-2x-DGX-Sparks" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND Apache-2.0 AND MIT" \
      recipe.snapshot="2026-10-19" recipe.weights="not-included"
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/opt/recipe/runtime/serve.py"]
