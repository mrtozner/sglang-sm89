"""Live Block-FP8 [128, 128] Triton GEMM autotuner and cache linker for sm_89 GPUs."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BLOCK_M_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
MAX_ADA_SMEM_BYTES = 100 * 1024  # 100 KiB configurable SMEM cap per CTA on sm_89


@dataclass(frozen=True)
class ScheduleCandidate:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def estimated_smem_bytes(self) -> int:
        # Activations load as 1-byte FP8 (or 2-byte staging) + 1-byte FP8 weights per stage.
        a_tile = self.block_m * self.block_k
        b_tile = self.block_n * self.block_k
        return (a_tile + b_tile) * self.num_stages


def normalize_device_name(raw_name: str) -> str:
    return re.sub(r"\s+", "_", raw_name.strip())


def get_config_cache_dir() -> Path:
    env_dir = os.getenv("SGLANG_SM89_CONFIG_CACHE", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".cache" / "sglang-sm89" / "fp8_configs"


def find_sglang_gemm_config_dir() -> Optional[Path]:
    env_dir = os.getenv("SGLANG_SM89_FP8_CONFIG_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir():
            return p

    for mod_name in (
        "sglang.kernels.ops.gemm",
        "sglang.srt.layers.quantization.fp8_kernel",
    ):
        try:
            mod = __import__(mod_name, fromlist=["__file__"])
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                base = Path(mod_file).resolve().parent
                for cand in (
                    base / "configs",
                    base.parent.parent / "kernels" / "ops" / "gemm" / "configs",
                ):
                    if cand.is_dir():
                        return cand
        except Exception:
            continue

    for static_cand in (
        Path("/sgl-workspace/sglang/python/sglang/kernels/ops/gemm/configs"),
        Path("/opt/sglang/lib/python3.12/site-packages/sglang/kernels/ops/gemm/configs"),
    ):
        if static_cand.is_dir():
            return static_cand
    return None


def config_filename(n: int, k: int, device_name: str) -> str:
    return (
        f"N={n},K={k},device_name={normalize_device_name(device_name)},"
        "dtype=fp8_w8a8,block_shape=[128, 128].json"
    )


def enumerate_sm89_candidates(m_tokens: int) -> List[ScheduleCandidate]:
    """Generate Ada sm_89 SRAM-constrained Triton Block-FP8 schedule candidates for bucket M."""
    if m_tokens <= 16:
        m_opts = (16,)
        n_opts = (16, 32, 64)
        warp_opts = (2, 4)
        stage_opts = (3, 4, 5)
        group_opts = (1, 8)
    elif m_tokens <= 64:
        m_opts = (16, 32, 64)
        n_opts = (32, 64, 128)
        warp_opts = (4, 8)
        stage_opts = (3, 4, 5)
        group_opts = (1, 8)
    else:
        m_opts = (64, 128)
        n_opts = (64, 128)
        warp_opts = (4, 8)
        stage_opts = (2, 3, 4)
        group_opts = (8, 16)

    valid: List[ScheduleCandidate] = []
    for bm in m_opts:
        for bn in n_opts:
            for nw in warp_opts:
                for ns in stage_opts:
                    for gm in group_opts:
                        cand = ScheduleCandidate(
                            block_m=bm,
                            block_n=bn,
                            block_k=128,
                            group_m=gm,
                            num_warps=nw,
                            num_stages=ns,
                        )
                        if cand.estimated_smem_bytes() <= MAX_ADA_SMEM_BYTES:
                            valid.append(cand)
    return valid


def synthesize_analytical_sm89_schedule(
    n: int, k: int, *, sm_count: int = 114
) -> Dict[str, Dict[str, int]]:
    """Construct a legal Ada sm_89 analytical schedule without static pre-baked JSON tables."""
    schedule: Dict[str, Dict[str, int]] = {}
    for m_val in BLOCK_M_BUCKETS:
        if m_val <= 16:
            bm = 16
            bn = 32 if (n * k) < (64 * 1024 * 1024) else 64
            nw, ns, gm = (4, 4, 1) if sm_count >= 114 else (4, 3, 1)
        elif m_val <= 64:
            bm = 32 if m_val >= 32 else 16
            bn = 64
            nw, ns, gm = (4, 4, 8)
        elif m_val <= 256:
            bm = 64
            bn = 128 if n >= 5120 else 64
            nw, ns, gm = (4, 3, 8)
        else:
            bm = 128
            bn = 128
            nw, ns, gm = (8, 3, 8)

        cand = ScheduleCandidate(bm, bn, 128, gm, nw, ns)
        if cand.estimated_smem_bytes() > MAX_ADA_SMEM_BYTES:
            cand = ScheduleCandidate(min(bm, 64), min(bn, 128), 128, gm, nw, 2)
        schedule[str(m_val)] = cand.to_dict()
    return schedule


def derive_fp8_shapes_from_config(cfg: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Extract unique [128, 128]-divisible Block-FP8 GEMM (N, K) shapes from an HF config dict."""
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    hidden = int(tc.get("hidden_size", 0))
    if hidden <= 0:
        raise ValueError("Model config missing positive hidden_size")

    inter = int(tc.get("intermediate_size", 0))
    moe_inter = int(tc.get("moe_intermediate_size", 0))
    num_heads = int(tc.get("num_attention_heads", 0))
    num_kv = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim", hidden // max(num_heads, 1)))

    shapes: List[Tuple[int, int]] = []
    if num_heads > 0 and head_dim > 0:
        q_dim = num_heads * head_dim
        kv_dim = num_kv * head_dim
        # Fused QKV and output projection
        shapes.append((q_dim + 2 * kv_dim, hidden))
        shapes.append((hidden, q_dim))

    # Gated DeltaNet linear attention shapes if present
    gdn_kh = int(tc.get("linear_num_key_heads", 0))
    gdn_vh = int(tc.get("linear_num_value_heads", 0))
    gdn_kd = int(tc.get("linear_key_head_dim", 0))
    gdn_vd = int(tc.get("linear_value_head_dim", 0))
    if gdn_kh > 0 and gdn_vh > 0 and gdn_kd > 0 and gdn_vd > 0:
        qkv_gdn = 2 * gdn_kh * gdn_kd + gdn_vh * gdn_vd
        z_dim = gdn_vh * gdn_vd
        shapes.append((qkv_gdn + z_dim, hidden))
        shapes.append((hidden, z_dim))

    mlp_dim = moe_inter if moe_inter > 0 else inter
    if mlp_dim > 0:
        shapes.append((2 * mlp_dim, hidden))
        shapes.append((hidden, mlp_dim))

    seen: set[Tuple[int, int]] = set()
    unique_shapes: List[Tuple[int, int]] = []
    for n_val, k_val in shapes:
        if n_val > 0 and k_val > 0 and n_val % 128 == 0 and k_val % 128 == 0:
            if (n_val, k_val) not in seen:
                seen.add((n_val, k_val))
                unique_shapes.append((n_val, k_val))
    return unique_shapes


def load_hf_config_dict(model_or_path: str) -> Dict[str, Any]:
    local_p = Path(model_or_path)
    if local_p.is_file():
        return json.loads(local_p.read_text(encoding="utf-8"))
    if local_p.is_dir() and (local_p / "config.json").is_file():
        return json.loads((local_p / "config.json").read_text(encoding="utf-8"))

    try:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_or_path, "config.json")
        return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    except Exception:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_or_path).to_dict()


def _resolve_block_fp8_matmul():
    for mod_name in (
        "sglang.kernels.ops.gemm.fp8_kernel",
        "sglang.srt.layers.quantization.fp8_kernel",
    ):
        try:
            mod = __import__(mod_name, fromlist=["w8a8_block_fp8_matmul_triton", "w8a8_block_fp8_matmul"])
            for fn_name in ("w8a8_block_fp8_matmul_triton", "w8a8_block_fp8_matmul"):
                fn = getattr(mod, fn_name, None)
                if fn is not None:
                    return mod, fn
        except Exception:
            continue
    return None, None


def autotune_shape_on_gpu(
    n: int,
    k: int,
    *,
    warmup_ms: int = 10,
    rep_ms: int = 40,
    allow_fallback: bool = False,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Benchmark Block-FP8 [128, 128] Triton schedules on the active CUDA GPU."""
    import inspect
    import torch
    import triton.testing

    if not torch.cuda.is_available():
        raise RuntimeError("GPU autotuning requires an active CUDA device")

    mod, matmul_fn = _resolve_block_fp8_matmul()
    if mod is None or matmul_fn is None:
        raise RuntimeError(
            "SGLang w8a8_block_fp8_matmul_triton kernel not found in current Python environment"
        )

    device = torch.device("cuda", 0)
    k_blocks = k // 128
    n_blocks = n // 128
    b_w = (torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1).to(
        torch.float8_e4m3fn
    )
    bs_scale = torch.ones((n_blocks, k_blocks), device=device, dtype=torch.float32)

    loader_attr = None
    orig_loader = None
    for cand_attr in ("get_w8a8_block_fp8_configs", "_get_w8a8_block_fp8_configs"):
        fn_obj = getattr(mod, cand_attr, None)
        if callable(fn_obj):
            loader_attr = cand_attr
            orig_loader = fn_obj
            break

    sig = inspect.signature(matmul_fn)
    accepts_config = "config" in sig.parameters
    if loader_attr is None and not accepts_config:
        raise RuntimeError(
            f"Cannot inject candidate schedules into {mod.__name__}.{matmul_fn.__name__}"
        )

    best_schedule: Dict[str, Dict[str, int]] = {}
    tuned_buckets = 0
    fallback_buckets = 0

    for m_val in BLOCK_M_BUCKETS:
        a = (torch.randn((m_val, k), device=device, dtype=torch.bfloat16) * 0.1).to(
            torch.float8_e4m3fn
        )
        as_scale = torch.ones((m_val, k_blocks), device=device, dtype=torch.float32)

        best_ms = float("inf")
        best_cand: Optional[ScheduleCandidate] = None
        last_exc: Optional[Exception] = None

        for cand in enumerate_sm89_candidates(m_val):
            cand_dict = cand.to_dict()
            if loader_attr is not None:
                fake_cfg = {m_val: cand_dict}
                setattr(mod, loader_attr, lambda *_a, _fc=fake_cfg, **_kw: _fc)
                cache_clear = getattr(orig_loader, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()
            try:
                def _run():
                    if accepts_config:
                        matmul_fn(
                            a,
                            b_w,
                            as_scale,
                            bs_scale,
                            [128, 128],
                            output_dtype=torch.bfloat16,
                            config=cand_dict,
                        )
                    else:
                        matmul_fn(
                            a,
                            b_w,
                            as_scale,
                            bs_scale,
                            [128, 128],
                            output_dtype=torch.bfloat16,
                        )

                # Correctness/compilation smoke before timing
                _run()
                torch.cuda.synchronize(device)
                elapsed_ms = float(
                    triton.testing.do_bench(_run, warmup=warmup_ms, rep=rep_ms)
                )
                if elapsed_ms < best_ms:
                    best_ms = elapsed_ms
                    best_cand = cand
            except Exception as exc:
                last_exc = exc
                continue

        if best_cand is None:
            if not allow_fallback:
                raise RuntimeError(
                    f"No candidate succeeded for shape ({n}, {k}) M={m_val}: {last_exc}"
                )
            fallback_buckets += 1
            best_schedule[str(m_val)] = synthesize_analytical_sm89_schedule(n, k)[str(m_val)]
        else:
            tuned_buckets += 1
            best_schedule[str(m_val)] = best_cand.to_dict()

        del a, as_scale

    if loader_attr is not None and orig_loader is not None:
        setattr(mod, loader_attr, orig_loader)
        cache_clear = getattr(orig_loader, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

    if tuned_buckets == 0:
        raise RuntimeError(f"Live GPU autotuning produced 0 tuned buckets for ({n}, {k})")

    stats = {"tuned_buckets": tuned_buckets, "fallback_buckets": fallback_buckets}
    return best_schedule, stats


def _link_or_copy(src: Path, dst: Path, *, force: bool) -> bool:
    if dst.exists() or dst.is_symlink():
        if not force:
            return False
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def _detect_sm89_device() -> Optional[Tuple[str, int]]:
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            return None
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (8, 9):
            return None
        props = torch.cuda.get_device_properties(0)
        return normalize_device_name(props.name), int(props.multi_processor_count)
    except Exception:
        return None


def ensure_sm89_fp8_configs_installed(
    *,
    cache_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    device_name: Optional[str] = None,
    force: bool = False,
) -> int:
    """Link cached autotuned FP8 JSONs (or synthesize analytical configs if explicitly enabled)."""
    dest_dir = target_dir if target_dir is not None else find_sglang_gemm_config_dir()
    if dest_dir is None:
        return 0

    src_cache = cache_dir if cache_dir is not None else get_config_cache_dir()
    installed = 0

    # 1. Link any user-autotuned JSONs from cache_dir matching the live/target device name.
    if src_cache.is_dir():
        det = _detect_sm89_device()
        active_dev = normalize_device_name(device_name) if device_name else (det[0] if det else None)
        for json_file in sorted(src_cache.glob("*.json")):
            if active_dev and f"device_name={active_dev}," not in json_file.name:
                continue
            dest_file = dest_dir / json_file.name
            try:
                if _link_or_copy(json_file, dest_file, force=force):
                    installed += 1
            except PermissionError:
                logger.warning("[sglang_sm89] Read-only SGLang config dir %s", dest_dir)
                return installed

    # 2. Optional analytical Ada schedule fallback when SGLANG_SM89_SYNTHESIZE_UNTUNED_FP8=1
    if installed == 0 and os.getenv("SGLANG_SM89_SYNTHESIZE_UNTUNED_FP8", "0").strip() == "1":
        det = _detect_sm89_device()
        dev_str = normalize_device_name(device_name) if device_name else (det[0] if det else None)
        sm_count = det[1] if det else 114
        shapes_env = os.getenv(
            "SGLANG_SM89_ANALYTICAL_SHAPES",
            "14336x5120,16384x5120,34816x5120,5120x17408,5120x6144",
        )
        if dev_str:
            src_cache.mkdir(parents=True, exist_ok=True)
            for spec in parse_shapes_arg(shapes_env):
                fname = config_filename(spec[0], spec[1], dev_str)
                cache_path = src_cache / fname
                if not cache_path.exists() or force:
                    sched = synthesize_analytical_sm89_schedule(
                        spec[0], spec[1], sm_count=sm_count
                    )
                    cache_path.write_text(json.dumps(sched, indent=4) + "\n", encoding="utf-8")
                dest_file = dest_dir / fname
                try:
                    if _link_or_copy(cache_path, dest_file, force=force):
                        installed += 1
                except PermissionError:
                    return installed

    if installed > 0:
        logger.info("[sglang_sm89] Linked %d Block-FP8 configs into %s", installed, dest_dir)
    return installed


def parse_shapes_arg(shapes_str: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for raw in shapes_str.split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"Invalid shape '{raw}'; expected NxK (e.g. 14336x5120)")
        n_s, k_s = tok.split("x", 1)
        out.append((int(n_s), int(k_s)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m sglang_sm89.autotune_fp8",
        description="Live GPU Block-FP8 [128, 128] Triton GEMM autotuner for sm_89",
    )
    parser.add_argument("--model", type=str, default=None, help="HF model ID or local config dir")
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated NxK shapes (e.g. 14336x5120,16384x5120)",
    )
    parser.add_argument(
        "--cache-dir",
        "--out",
        dest="cache_dir",
        type=Path,
        default=None,
        help="Output JSON cache dir",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print shapes/analytical config only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached configs")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow analytical fallback if an individual M bucket fails",
    )
    args = parser.parse_args(argv)

    shapes: List[Tuple[int, int]] = []
    if args.model:
        cfg = load_hf_config_dict(args.model)
        shapes.extend(derive_fp8_shapes_from_config(cfg))
    if args.shapes:
        for s in parse_shapes_arg(args.shapes):
            if s not in shapes:
                shapes.append(s)

    if not shapes:
        parser.error("Provide --model and/or --shapes")

    cache_dir = args.cache_dir if args.cache_dir is not None else get_config_cache_dir()
    det = _detect_sm89_device()
    dev_name = det[0] if det else "NVIDIA_GeForce_RTX_4090_D"

    if args.dry_run:
        print(json.dumps({"device": dev_name, "shapes": shapes, "cache_dir": str(cache_dir)}, indent=2))
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    for n_val, k_val in shapes:
        out_path = cache_dir / config_filename(n_val, k_val, dev_name)
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path}")
            continue
        schedule, stats = autotune_shape_on_gpu(
            n_val, k_val, allow_fallback=args.allow_fallback
        )
        out_path.write_text(json.dumps(schedule, indent=4) + "\n", encoding="utf-8")
        print(
            f"[tuned] {out_path} (tuned_buckets={stats['tuned_buckets']}/{len(BLOCK_M_BUCKETS)}, "
            f"fallback_buckets={stats['fallback_buckets']})"
        )

    linked = ensure_sm89_fp8_configs_installed(cache_dir=cache_dir, device_name=dev_name, force=True)
    print(f"[linked] {linked} config(s) into SGLang GEMM config directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
