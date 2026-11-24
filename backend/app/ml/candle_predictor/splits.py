"""
Validación cruzada cronológica Multi-Ticker (Walk-Forward Expansiva con Embargo)
sobre línea de tiempo UTC global, sin solapamiento de ventanas ni look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.app.ml.candle_predictor.features import TickerFrame, build_index

logger = logging.getLogger(__name__)


def _ns_to_iso(ns: Optional[int]) -> str:
    if ns is None:
        return ""
    return datetime.fromtimestamp(float(ns) / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@dataclass
class WalkForwardFold:
    fold: int
    train_index: List[Tuple[int, int]]
    val_index: List[Tuple[int, int]]
    train_max_end_by_ticker: Dict[int, int]
    val_start_ns: int
    val_end_ns: int


@dataclass
class WalkForwardPlan:
    folds: List[WalkForwardFold]
    final_train_index: List[Tuple[int, int]]
    final_cal_index: List[Tuple[int, int]]
    final_te_index: List[Tuple[int, int]]
    final_train_max_end_by_ticker: Dict[int, int]
    te_start_ns: int
    cal_start_ns: int
    embargo_bars: int

    def summary_dict(self) -> Dict[str, object]:
        return {
            "embargo_bars": int(self.embargo_bars),
            "cal_start_utc": _ns_to_iso(self.cal_start_ns),
            "te_start_utc": _ns_to_iso(self.te_start_ns),
            "num_folds_valid": len(self.folds),
            "folds": [
                {
                    "fold": f.fold,
                    "val_start_utc": _ns_to_iso(f.val_start_ns),
                    "val_end_utc": _ns_to_iso(f.val_end_ns),
                    "train_samples": len(f.train_index),
                    "val_samples": len(f.val_index),
                }
                for f in self.folds
            ],
            "final_train_samples": len(self.final_train_index),
            "final_cal_samples": len(self.final_cal_index),
            "final_te_samples": len(self.final_te_index),
        }


def _embargoed_train_before(
    frames: Sequence[TickerFrame],
    seq_len: int,
    embargo: int,
    next_start_ns: int,
    next_index: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Construye los índices de entrenamiento estrictamente previos a `next_start_ns`,
    respetando por ticker `t <= v0 - embargo - 2` (de modo que el label t+1 quede
    al menos `embargo` velas antes de `v0`) y, cuando `embargo >= seq_len`, forzando
    también la cota global `ts[t+1] < min(ts[window_start de next_index])`.
    """
    if next_index:
        min_next_win_start_ns = min(
            int(frames[ti].ts[ei - seq_len + 1]) for ti, ei in next_index
        )
    else:
        min_next_win_start_ns = next_start_ns

    global_label_ceil_ns = (
        min(next_start_ns, min_next_win_start_ns)
        if embargo >= seq_len
        else next_start_ns
    )

    train_pairs: List[Tuple[int, int]] = []
    max_end_by_ticker: Dict[int, int] = {}

    for t_idx, frame in enumerate(frames):
        n = len(frame.X)
        if n < seq_len + 1:
            continue
        v0 = int(np.searchsorted(frame.ts, next_start_ns, side="left"))
        if v0 < n:
            max_allowed = min(v0 - int(embargo) - 2, n - 2)
        else:
            # Todas las velas de este ticker terminan antes de `next_start_ns`
            max_allowed = n - 2

        if max_allowed < seq_len - 1:
            continue

        last_valid_t = -1
        for end_idx in range(seq_len - 1, max_allowed + 1):
            label_ts = int(frame.ts[end_idx + 1])
            if label_ts >= global_label_ceil_ns:
                break
            start_idx = end_idx - seq_len + 1
            if not bool(frame.valid[start_idx : end_idx + 1].all()):
                continue
            if not (
                np.isfinite(frame.y_ret[end_idx])
                and np.isfinite(frame.y_tr[end_idx])
                and np.isfinite(frame.y_dir[end_idx])
            ):
                continue
            train_pairs.append((t_idx, end_idx))
            last_valid_t = end_idx

        if last_valid_t >= 0:
            max_end_by_ticker[t_idx] = last_valid_t

    # Orden cronológico global sin shuffle: (timestamp_fin, ticker_idx, end_idx)
    train_pairs.sort(key=lambda pair: (int(frames[pair[0]].ts[pair[1]]), pair[0], pair[1]))
    return train_pairs, max_end_by_ticker


def make_walk_forward_splits(
    frames: Sequence[TickerFrame],
    seq_len: int,
    n_folds: int = 4,
    te_frac: float = 0.15,
    embargo: int = 24,
    min_fold_train: int = 200,
    min_fold_val: int = 50,
) -> WalkForwardPlan:
    """
    Construye divisiones cronológicas Walk-Forward expansivas con embargo sobre
    la línea de tiempo UTC global unificada de todos los tickers.
    """
    if not frames:
        raise ValueError("Se requiere al menos un TickerFrame para construir splits.")

    all_ts_parts: List[np.ndarray] = []
    for frame in frames:
        n = len(frame.X)
        if n < seq_len:
            continue
        ok = np.isfinite(frame.y_ret) & np.isfinite(frame.y_tr) & np.isfinite(frame.y_dir) & frame.valid
        valid_ends = np.flatnonzero(ok[seq_len - 1 :]) + (seq_len - 1)
        if len(valid_ends) > 0:
            all_ts_parts.append(frame.ts[valid_ends])

    if not all_ts_parts:
        raise ValueError("No hay muestras válidas con targets finitos. Use --source yfinance.")

    timeline = np.unique(np.concatenate(all_ts_parts))
    if len(timeline) < (n_folds + 2) * 4:
        raise ValueError(
            f"Línea temporal demasiado corta ({len(timeline)} puntos únicos) para Walk-Forward "
            f"con {n_folds} pliegues y embargo={embargo}. Ejecute con --source yfinance."
        )

    te_idx = int(np.floor(len(timeline) * (1.0 - float(te_frac))))
    te_idx = max(n_folds + 1, min(len(timeline) - 2, te_idx))
    te_start_ns = int(timeline[te_idx])

    dev_timeline = timeline[:te_idx]
    block_edges = np.linspace(0, len(dev_timeline), int(n_folds) + 2, dtype=int)

    folds: List[WalkForwardFold] = []
    for k in range(1, int(n_folds) + 1):
        val_start_pos = int(block_edges[k])
        val_end_pos = int(block_edges[k + 1])
        val_start_ns = int(dev_timeline[val_start_pos])
        val_end_ns = (
            int(dev_timeline[val_end_pos]) if val_end_pos < len(dev_timeline) else te_start_ns
        )

        val_idx = build_index(
            frames=frames,
            seq_len=seq_len,
            ts_lo=val_start_ns,
            ts_hi=val_end_ns,
            require_target=True,
        )
        val_idx.sort(key=lambda pair: (int(frames[pair[0]].ts[pair[1]]), pair[0], pair[1]))

        train_idx, max_end_map = _embargoed_train_before(
            frames=frames,
            seq_len=seq_len,
            embargo=embargo,
            next_start_ns=val_start_ns,
            next_index=val_idx,
        )

        if len(train_idx) < min_fold_train or len(val_idx) < min_fold_val:
            logger.warning(
                "Fold %d omitido por tamaño insuficiente (train=%d<%d o val=%d<%d).",
                k,
                len(train_idx),
                min_fold_train,
                len(val_idx),
                min_fold_val,
            )
            continue

        folds.append(
            WalkForwardFold(
                fold=k,
                train_index=train_idx,
                val_index=val_idx,
                train_max_end_by_ticker=max_end_map,
                val_start_ns=val_start_ns,
                val_end_ns=val_end_ns,
            )
        )

    if not folds:
        raise ValueError(
            "Ningún pliegue Walk-Forward alcanzó el mínimo de muestras con embargo="
            f"{embargo}. Use --source yfinance para descargar histórico extendido."
        )

    # Split final: Train (bloques 0..n_folds-1) | Embargo | Calibración (último bloque dev) | Embargo | TE
    cal_start_ns = int(dev_timeline[int(block_edges[-2])])

    te_idx_list = build_index(
        frames=frames,
        seq_len=seq_len,
        ts_lo=te_start_ns,
        ts_hi=None,
        require_target=True,
    )
    te_idx_list.sort(key=lambda pair: (int(frames[pair[0]].ts[pair[1]]), pair[0], pair[1]))

    # Embargamos el extremo final de calibración respecto a TE igual que train respecto a cal
    raw_cal_all, _ = _embargoed_train_before(
        frames=frames,
        seq_len=seq_len,
        embargo=embargo,
        next_start_ns=te_start_ns,
        next_index=te_idx_list,
    )
    cal_idx_list = [
        (ti, ei) for (ti, ei) in raw_cal_all if int(frames[ti].ts[ei]) >= cal_start_ns
    ]
    cal_idx_list.sort(key=lambda pair: (int(frames[pair[0]].ts[pair[1]]), pair[0], pair[1]))

    final_train_idx, final_max_end_map = _embargoed_train_before(
        frames=frames,
        seq_len=seq_len,
        embargo=embargo,
        next_start_ns=cal_start_ns,
        next_index=cal_idx_list if cal_idx_list else te_idx_list,
    )

    if len(final_train_idx) < min_fold_train or len(cal_idx_list) < min_fold_val or not te_idx_list:
        raise ValueError(
            f"Split final insuficiente (train={len(final_train_idx)}, cal={len(cal_idx_list)}, "
            f"te={len(te_idx_list)}). Use --source yfinance para histórico completo."
        )

    return WalkForwardPlan(
        folds=folds,
        final_train_index=final_train_idx,
        final_cal_index=cal_idx_list,
        final_te_index=te_idx_list,
        final_train_max_end_by_ticker=final_max_end_map,
        te_start_ns=te_start_ns,
        cal_start_ns=cal_start_ns,
        embargo_bars=int(embargo),
    )
