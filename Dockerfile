FROM lmsysorg/sglang@sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652

LABEL org.opencontainers.image.title="sglang-sm89" \
      org.opencontainers.image.description="SGLang v0.5.23 with Ada Lovelace (sm_89) live FP8 autotuner, GDN ReplaySSM WY verify, coupled grammar resampling, and kTileV=64 sparse vocab projection" \
      org.opencontainers.image.source="https://github.com/mrtozner/sglang-sm89" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV SGLANG_SM89_ENABLE=1 \
    SGLANG_SM89_FP8_CONFIGS=1 \
    SGLANG_SM89_SYNTHESIZE_UNTUNED_FP8=0 \
    SGLANG_SM89_DFLASH_FP8=1 \
    SGLANG_SM89_REPLAYSSM=1 \
    SGLANG_SM89_COUPLED_GRAMMAR=1 \
    SGLANG_SM89_SPARSE_PROJ=1 \
    SGLANG_SM89_WY_GDN=1 \
    SGLANG_SM89_WY_GDN_MAX_BF16_TOKENS=32 \
    SGLANG_SM89_FP8_VERIFY_HEAD=0 \
    SGLANG_SM89_SAM_GRAFT=0

WORKDIR /opt/sglang-sm89
COPY . /opt/sglang-sm89

RUN python3 -m pip install --no-cache-dir --no-build-isolation -v . && \
    (cd /tmp && python3 -c "import os, pathlib, torch, sglang_sm89._C as c; src = pathlib.Path(c.__file__).resolve(); dst = pathlib.Path('/opt/sglang-sm89/sglang_sm89') / src.name; dst.unlink(missing_ok=True); os.symlink(src, dst); print('Linked', dst, '->', src)") && \
    ([ -e /app ] || ln -s /opt/sglang-sm89 /app) && \
    (cd /tmp && python3 -c "import torch, sglang_sm89, sglang_sm89._C; from sglang_sm89.autotune_fp8 import ensure_sm89_fp8_configs_installed; n = ensure_sm89_fp8_configs_installed(); print(f'sglang_sm89 ready (configs linked: {n})')") && \
    python3 -c "import torch, sglang_sm89, sglang_sm89._C; print('WORKDIR import OK:', sglang_sm89._C.__file__)"
