"""Audit complete Spark replication and summarize sampled metrics, never invent N/A."""
import argparse
import json
import math
from pathlib import Path


def numeric(v):
    try:
        n = float(v)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def energy(samples, start, stop):
    """Trapezoidal integration with interpolated endpoints; require full coverage."""
    import numpy as np
    points = [(r["monotonic"], numeric(r["devices"][0].get("power.draw")) if len(r["devices"]) == 1 else None)
              for r in samples]
    # Missing samples inside an interval must not silently disappear.
    left = [p for p in points if p[0] <= start]
    right = [p for p in points if p[0] >= stop]
    if not left or not right:
        return None
    selected = [left[-1]] + [p for p in points if start < p[0] < stop] + [right[0]]
    if any(p[1] is None for p in selected) or any(b[0]-a[0] > 3 for a,b in zip(selected,selected[1:])):
        return None
    ts, ps = zip(*selected)
    x = [start] + [t for t in ts if start < t < stop] + [stop]
    y = np.interp(x, ts, ps)
    return float(sum((b-a)*(u+v)/2 for a,b,u,v in zip(x,x[1:],y,y[1:])))

