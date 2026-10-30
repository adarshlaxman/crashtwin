"""Evaluation metrics (EXPERIMENT_DESIGN.md section 7). Pure numpy/pandas.

ccd_protocol   : AP, mTTA, TTA@R80 exactly as in the UString/DSTA evaluation code (comparable
                 to published CCD numbers; its mTTA is relative to the clip length).
tta_seconds    : our direct warning time: mean over true-positive videos of
                 (event time - first alarm time), at the validation threshold.
nexar_official : Nexar's score(): mean AP over the time_to_accident groups (one score per video).
fa_per_minute  : alarm onsets per minute on negative videos at a threshold.
bootstrap_ci   : 95 % CI over videos."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def ccd_protocol(preds, labels, toas, n_frames_total, fps=10.0):
    """preds: list of per-frame score arrays; labels: 0/1; toas: event frame index (positives).
    Faithful port of UString `evaluation()` (Bao et al. 2020)."""
    preds_eval, min_pred = [], np.inf
    for p, y, toa in zip(preds, labels, toas):
        p = np.asarray(p, float)
        p = p[:int(toa)] if y > 0 else p
        if len(p) == 0:
            p = np.zeros(1)
        min_pred = min(min_pred, p.min())
        preds_eval.append(p)
    labels = np.asarray(labels)
    total_seconds = n_frames_total / fps
    P, R, Tm = [], [], []
    for th in np.arange(max(min_pred, 0), 1.0, 0.001):
        tp = tp_fp = time = counter = 0.0
        for p, y, toa in zip(preds_eval, labels, toas):
            hit = np.where(p * y >= th)[0]
            if len(hit) > 0:
                tp += 1
                time += hit[0] / float(toa)
                counter += 1
            tp_fp += float((p >= th).any())
        if tp_fp == 0 or labels.sum() == 0 or counter == 0:
            continue
        P.append(tp / tp_fp)
        R.append(tp / labels.sum())
        Tm.append(1 - time / counter)
    if not R:
        return {"AP": 0.0, "mTTA": 0.0, "TTA_R80": 0.0}
    P, R, Tm = map(np.array, (P, R, Tm))
    order = np.argsort(R)
    P, R, Tm = P[order], R[order], Tm[order]
    _, rep = np.unique(R, return_index=True)
    rep = rep[1:] if len(rep) > 1 else rep
    newP, newT = np.zeros(len(rep)), np.zeros(len(rep))
    for i in range(len(rep) - 1):
        newT[i] = np.max(Tm[rep[i]:rep[i + 1]])
        newP[i] = np.max(P[rep[i]:rep[i + 1]])
    newT[-1], newP[-1] = Tm[rep[-1]], P[rep[-1]]
    newR = R[rep]
    ap = newP[0] * newR[0] if newR[0] != 0 else 0.0
    for i in range(1, len(newP)):
        ap += (newP[i - 1] + newP[i]) * (newR[i] - newR[i - 1]) / 2
    mtta = float(np.mean(newT) * total_seconds)
    idx = np.where(newR >= 0.8)[0]
    tta80 = float(newT[idx[0]] * total_seconds) if len(idx) else float("nan")
    return {"AP": float(ap), "mTTA": mtta, "TTA_R80": tta80}


def first_alarm(p, th):
    hit = np.where(np.asarray(p) >= th)[0]
    return int(hit[0]) if len(hit) else None


def tta_seconds(preds, labels, toas, th, fps=10.0):
    """Mean warning time (s) over positives warned before the event, and video-level recall."""
    tt, warned, pos = [], 0, 0
    for p, y, toa in zip(preds, labels, toas):
        if y != 1:
            continue
        pos += 1
        f = first_alarm(np.asarray(p)[:int(toa)], th)
        if f is not None:
            warned += 1
            tt.append((toa - f) / fps)
    return {"tta_s": float(np.mean(tt)) if tt else 0.0, "recall": warned / max(pos, 1)}


def threshold_for_recall(preds, labels, toas, target=0.8):
    """Highest threshold that still warns before the event on >= target of positives."""
    peaks = sorted([float(np.max(np.asarray(p)[:int(toa)])) if int(toa) > 0 else 0.0
                    for p, y, toa in zip(preds, labels, toas) if y == 1], reverse=True)
    if not peaks:
        return 0.5
    k = max(0, int(np.ceil(target * len(peaks))) - 1)
    return peaks[k]


def fa_per_minute(preds, labels, th, fps=10.0):
    onsets, frames = 0, 0
    for p, y in zip(preds, labels):
        if y != 0:
            continue
        a = np.asarray(p) >= th
        onsets += int(a[0]) + int(np.sum(a[1:] & ~a[:-1]))
        frames += len(a)
    return onsets / max(frames / fps / 60.0, 1e-9)


def nexar_official(scores, solution_csv):
    """scores: dict video id (e.g. '00002') -> score. Returns {'Public': mAP, 'Private': mAP}."""
    sol = pd.read_csv(solution_csv, dtype={"id": str})
    sub = pd.DataFrame({"id": list(scores), "score": list(scores.values())})
    out = {}
    for usage in ("Public", "Private"):
        s = sol[sol["Usage"] == usage].merge(sub, on="id", how="left").fillna({"score": 0})
        out[usage] = float(np.mean([average_precision_score(s.loc[s.group == g, "target"],
                                                            s.loc[s.group == g, "score"])
                                    for g in s.group.unique()]))
    return out


def bootstrap_ci(fn, n_items, n=1000, seed=0):
    """95 % CI of fn(idx) over resampled item indices."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)
