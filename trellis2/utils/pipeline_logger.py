"""
Pipeline logging utility for Trellis 2.
Writes to both /tmp/trellis2_pipeline.log (full DEBUG) and stdout (INFO+).
Call reset_log() at the start of each new generation run.
"""

import sys
import time
import logging
import traceback
from datetime import datetime
from typing import Optional

import torch
import numpy as np

LOG_PATH = "/tmp/trellis2_pipeline.log"

_logger: Optional[logging.Logger] = None
_run_start: float = 0.0
_debug_enabled: bool = False


def _make_logger() -> logging.Logger:
    log = logging.getLogger("trellis2_pipeline")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if _debug_enabled else logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    return log


def reset_log(label: str = "") -> None:
    """Call at the very start of every new generation request."""
    global _logger, _run_start
    _logger = _make_logger()
    _run_start = time.perf_counter()
    _logger.info("=" * 80)
    _logger.info(f"NEW PIPELINE RUN  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {label}")
    _logger.info("=" * 80)


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = _make_logger()
    return _logger


def set_debug(enabled: bool) -> None:
    """Enable or disable DEBUG-level output to stdout."""
    global _debug_enabled, _logger
    _debug_enabled = enabled
    # Update any already-created logger's stdout handler
    if _logger is not None:
        for handler in _logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG if enabled else logging.INFO)


def elapsed() -> str:
    return f"+{time.perf_counter() - _run_start:.2f}s"


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _ts(t) -> str:
    """One-line tensor summary."""
    if t is None:
        return "None"
    if not isinstance(t, torch.Tensor):
        return f"<{type(t).__name__}>"
    try:
        f = t.detach().float()
        has_nan = torch.isnan(f).any().item()
        has_inf = torch.isinf(f).any().item()
        if has_nan or has_inf:
            flags = ("NaN " if has_nan else "") + ("inf" if has_inf else "")
            return f"shape={list(t.shape)} dtype={t.dtype}  ⚠ {flags.strip()}"
        mn, mx = f.min().item(), f.max().item()
        mu = f.mean().item()
        return (f"shape={list(t.shape)} dtype={t.dtype}  "
                f"min={mn:.4g}  max={mx:.4g}  mean={mu:.4g}")
    except Exception as e:
        return f"shape={list(t.shape)} dtype={t.dtype}  [stats error: {e}]"


def log_tensor(t, name: str, level: str = "info") -> None:
    getattr(get_logger(), level)(f"  {elapsed()} [{name}] {_ts(t)}")


def log_mesh(vertices, faces, tag: str = "mesh") -> None:
    L = get_logger()
    prefix = f"  {elapsed()} [MESH:{tag}]"
    if vertices is None or faces is None:
        L.warning(f"{prefix} vertices or faces is None!")
        return
    try:
        v = vertices.detach().float() if isinstance(vertices, torch.Tensor) else torch.tensor(vertices, dtype=torch.float32)
        f = faces.detach() if isinstance(faces, torch.Tensor) else torch.tensor(faces)

        has_nan = torch.isnan(v).any().item()
        has_inf = torch.isinf(v).any().item()
        ok = "✅" if not has_nan and not has_inf else "❌"

        L.info(f"{prefix} {ok}  "
               f"vertices={list(v.shape)}  faces={list(f.shape)}  "
               f"pos=[{v.min().item():.4g}, {v.max().item():.4g}]  "
               f"NaN={has_nan}  inf={has_inf}")

        if f.numel() > 0:
            idx_min = int(f.min().item())
            idx_max = int(f.max().item())
            n_verts = v.shape[0]
            valid = (idx_min >= 0) and (idx_max < n_verts)
            flag = "✅" if valid else "❌ OUT-OF-BOUNDS"
            L.info(f"{prefix}   face-idx range=[{idx_min}, {idx_max}]  "
                   f"num_vertices={n_verts}  {flag}")
            if not valid:
                L.error(f"{prefix}   ⚠ INVALID FACE INDICES — expect corruption downstream!")

        if v.shape[0] >= 3:
            L.debug(f"{prefix}   first 3 verts: {v[:3].tolist()}")
        if f.shape[0] >= 3:
            L.debug(f"{prefix}   first 3 faces: {f[:3].tolist()}")

    except Exception as e:
        L.error(f"{prefix} exception: {e}\n{traceback.format_exc()}")


def log_uv(uv, tag: str = "uv") -> None:
    L = get_logger()
    prefix = f"  {elapsed()} [UV:{tag}]"
    if uv is None:
        L.warning(f"{prefix} None!")
        return
    try:
        t = uv.detach().float() if isinstance(uv, torch.Tensor) else torch.tensor(uv, dtype=torch.float32)
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        ok = "✅" if not has_nan and not has_inf else "❌"
        u_range = f"[{t[:, 0].min().item():.4g}, {t[:, 0].max().item():.4g}]"
        v_range = f"[{t[:, 1].min().item():.4g}, {t[:, 1].max().item():.4g}]"
        n_zero = (t.abs().sum(dim=-1) < 1e-8).sum().item()
        pct_zero = 100.0 * n_zero / max(1, t.shape[0])
        L.info(f"{prefix} {ok}  shape={list(t.shape)}  "
               f"U={u_range}  V={v_range}  "
               f"zeros={n_zero}/{t.shape[0]} ({pct_zero:.1f}%)  "
               f"NaN={has_nan}  inf={has_inf}")
        if pct_zero > 50:
            L.error(f"{prefix}  ⚠ >50% UV coordinates are zero — likely UV gen failure!")
        if t.shape[0] >= 3:
            L.debug(f"{prefix}  first 5 UVs: {t[:5].tolist()}")
    except Exception as e:
        L.error(f"{prefix} exception: {e}")


def log_sparse(sp_tensor, tag: str = "sparse") -> None:
    """Log a SparseTensor or VarLenTensor."""
    L = get_logger()
    prefix = f"  {elapsed()} [SPARSE:{tag}]"
    try:
        feats = sp_tensor.feats if hasattr(sp_tensor, "feats") else None
        coords = sp_tensor.coords if hasattr(sp_tensor, "coords") else None
        if feats is not None:
            L.info(f"{prefix}  feats: {_ts(feats)}")
        if coords is not None:
            L.info(f"{prefix}  coords: {_ts(coords)}  "
                   f"max={coords.max(dim=0).values.tolist() if coords.numel() > 0 else 'N/A'}")
    except Exception as e:
        L.error(f"{prefix} exception: {e}")


def section(title: str) -> None:
    L = get_logger()
    L.info("")
    L.info(f"{'─'*70}")
    L.info(f"  {elapsed()}  ▶  {title}")
    L.info(f"{'─'*70}")


def check_tensor(t, name: str, expect_finite: bool = True) -> bool:
    """Returns True if tensor passes checks. Logs ERROR if not."""
    if t is None:
        get_logger().warning(f"  {elapsed()} [{name}] is None")
        return False
    if not isinstance(t, torch.Tensor):
        return True
    f = t.detach().float()
    has_nan = torch.isnan(f).any().item()
    has_inf = torch.isinf(f).any().item()
    if expect_finite and (has_nan or has_inf):
        get_logger().error(f"  {elapsed()} [{name}] ❌ CORRUPT — NaN={has_nan}  inf={has_inf}  {_ts(t)}")
        return False
    return True
