"""
STKDE — densidade com kernel Gaussiano espaço-temporal.
"""

from __future__ import annotations

import numpy as np


def _t_eval_from_config(
    t_train: np.ndarray,
    t_test: np.ndarray,
    mode: str,
    fixed: float,
) -> float:
    mode = mode.lower()
    if mode == "median_test":
        return float(np.median(t_test))
    if mode == "max_train":
        return float(np.max(t_train))
    if mode == "fixed":
        return float(fixed)
    raise ValueError(f"t_eval_mode desconhecido: {mode}")


def stkde_density_at_points(
    train_xyt: np.ndarray,
    eval_xy: np.ndarray,
    t_eval: float,
    h_s: float,
    h_t: float,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    train_xyt: (n, 3) — x_utm, y_utm, t_horas
    eval_xy: (m, 2)
    Retorna lambda shape (m,).
    """
    if h_s <= 0 or h_t <= 0:
        raise ValueError("h_s e h_t devem ser positivos.")
    n = train_xyt.shape[0]
    if n == 0:
        raise ValueError("Treino vazio.")

    norm = (2.0 * np.pi * h_s**2) * (np.sqrt(2.0 * np.pi) * h_t)
    inv_2hs2 = 1.0 / (2.0 * h_s * h_s)
    inv_2ht2 = 1.0 / (2.0 * h_t * h_t)

    x0 = train_xyt[:, 0]
    y0 = train_xyt[:, 1]
    t0 = train_xyt[:, 2]

    m = eval_xy.shape[0]
    out = np.empty(m, dtype=np.float64)

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        x = eval_xy[start:end, 0][:, None]
        y = eval_xy[start:end, 1][:, None]
        dx = x - x0[None, :]
        dy = y - y0[None, :]
        d2 = dx * dx + dy * dy
        dt2 = (t_eval - t0[None, :]) ** 2
        k = np.exp(-d2 * inv_2hs2 - dt2 * inv_2ht2)
        out[start:end] = k.sum(axis=1) / (n * norm)

    return out


def loglik_test_events(
    train_xyt: np.ndarray,
    test_xyt: np.ndarray,
    h_s: float,
    h_t: float,
    *,
    chunk_size: int = 512,
    eps: float = 1e-12,
) -> float:
    """Soma log λ(x_j,y_j,t_j) nos eventos de teste (dentro da malha)."""
    if test_xyt.size == 0:
        return float("-inf")
    lam = stkde_density_at_points(
        train_xyt,
        test_xyt[:, :2],
        float(test_xyt[0, 2]),  # placeholder — sobrescrito abaixo
        h_s,
        h_t,
        chunk_size=chunk_size,
    )
    # Avaliar cada evento no seu próprio tempo (verossimilhança padrão)
    m = test_xyt.shape[0]
    ll = 0.0
    norm = (2.0 * np.pi * h_s**2) * (np.sqrt(2.0 * np.pi) * h_t)
    inv_2hs2 = 1.0 / (2.0 * h_s * h_s)
    inv_2ht2 = 1.0 / (2.0 * h_t * h_t)
    x0, y0, t0 = train_xyt[:, 0], train_xyt[:, 1], train_xyt[:, 2]

    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        block = test_xyt[start:end]
        x = block[:, 0][:, None]
        y = block[:, 1][:, None]
        t = block[:, 2][:, None]
        dx = x - x0[None, :]
        dy = y - y0[None, :]
        dt = t - t0[None, :]
        d2 = dx * dx + dy * dy
        k = np.exp(-d2 * inv_2hs2 - (dt * dt) * inv_2ht2)
        lam = k.sum(axis=1) / (train_xyt.shape[0] * norm)
        ll += float(np.log(lam + eps).sum())

    return ll