"""pytest plugin: plant one look-ahead selected by $PLANT (scratch only; no repo file is touched)."""

import os
import sys

import numpy as np
import pandas as pd


def _swap(orig, new):
    n = 0
    for m in list(sys.modules.values()):
        d = getattr(m, "__dict__", None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is orig:
                setattr(m, k, new)
                n += 1
    return n


def pytest_configure(config):
    plant = os.environ.get("PLANT", "")
    import features.groups, features.exits, features.vol_profile, features.events, features.session, wfo.rule_pass  # noqa
    import primaries.mechanism as M

    if plant == "FEAT":
        import dataclasses

        from features.registry import REGISTRY

        g = REGISTRY["volatility"]

        def leaky(df, cfg, context, _f=g.fn):
            out = _f(df, cfg, context)
            out["volatility__hl_range"] = out["volatility__hl_range"].shift(-1)
            return out

        REGISTRY["volatility"] = dataclasses.replace(g, fn=leaky)
    elif plant == "VOL":
        import features.vol_profile as V

        orig = V.bar_volatility

        def bv(close, span, profile=None):
            lr = np.log(close).diff().shift(-1)
            if profile is None:
                return lr.ewm(span=span, min_periods=span).std().rename("bar_vol")
            s = profile.factor(close.index)
            return ((lr / s).ewm(span=span, min_periods=span).std() * s).rename("bar_vol")

        print("swapped", _swap(orig, bv))
    elif plant == "EXIT_HYST":
        import features.exits as E

        orig = E.hysteresis_exits

        def hx(df, events, width, signal, side=None, *, beta, max_bars, hold_overnight):
            out = orig(df, events, width, signal, side, beta=beta, max_bars=max_bars, hold_overnight=hold_overnight)
            h = out["barrier"].to_numpy() == "hysteresis"
            x = out["exit_pos"].to_numpy().copy()
            x[h] -= 1  # fill at the OPEN of the trigger bar (decided at its close)
            px = out["exit_px"].to_numpy().copy()
            px[h] = df["open"].to_numpy()[x[h]]
            out["exit_pos"] = x
            out["exit_px"] = px
            out["t1"] = df.index[x]
            out["ret"] = out["exit_px"] / out["entry_px"] - 1
            out["label"] = np.sign(out["ret"]).astype(int)
            return out

        print("swapped", _swap(orig, hx))
    elif plant == "ENTRY":
        import features.triple_barrier_labels as T

        orig = T.barrier_exits

        def bx(df, events, width, side=None, *, vertical_bars, hold_overnight):
            # enter at the OPEN of the event bar itself (before the close that decides it)
            ev2 = df.index[np.maximum(df.index.get_indexer(events) - 1, 0)]
            w2 = pd.Series(width.reindex(events).to_numpy(), index=ev2)
            s2 = None if side is None else pd.Series(side.reindex(events).to_numpy(), index=ev2)
            out = orig(df, ev2, w2, s2, vertical_bars=vertical_bars, hold_overnight=hold_overnight)
            out.index = df.index[df.index.get_indexer(out.index) + 1]
            return out

        print("swapped", _swap(orig, bx))
    elif plant == "PRIM":
        orig = M.MechanismPrimary._at

        def at(self, s, df, X):
            pos = df.index.get_indexer(X.index)
            return np.asarray(s, dtype=float)[np.minimum(pos + 1, len(df) - 1)]  # value one bar after the event

        M.MechanismPrimary._at = at
    elif plant == "SESSION":
        import features.session as S

        orig = S.session_frame

        def sf(df, cfg, profile=None, iaom=None):
            out = orig(df, cfg, profile, iaom)
            out["open_to_now"] = out["open_to_now"].shift(-1)
            return out

        print("swapped", _swap(orig, sf))
    print(f"[PLANT] {plant or 'none'}")


def pytest_sessionstart(session):
    plant = os.environ.get("PLANT", "")
    if plant == "CAL":
        import wfo.wfo_engine as W

        orig = W.CalHistory.pairs

        def pairs(self, labels, fit_start, end, embargo, n_bars):
            # no purge on the label span: every stored pair with an event bar >= fit_start (outcomes may resolve later)
            return orig(self, labels, fit_start, n_bars + 10**9, 0, n_bars)

        W.CalHistory.pairs = pairs
        print("[PLANT] CAL")
