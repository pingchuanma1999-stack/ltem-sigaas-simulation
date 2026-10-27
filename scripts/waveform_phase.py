"""Waveform relation between two simulated detected traces, in the same way as the measured ±25 V analysis
(project notes, Oct 22): FFT of both traces, band 0.25-1.6 THz, fit phase(test) - phase(ref) = phi0 + 2 pi f tau.
phi0 = 0: same waveform (a pure delay shows only as a slope tau); 180 deg: inverted; +-90 deg: one is ~ the time
derivative of the other. Usage: python scripts/waveform_phase.py  (prints a table for the available runs)"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ltemsim.tip_demod import demodulate_positions, demodulate_pizzuto, detected_waveform

BAND = (0.25e12, 1.6e12)


def phase_fit(t, ref, test, band=BAND):
    """Returns phi0 (deg), tau (ps), rms phase residual (deg)."""
    n = len(t)
    pad = 4 * n
    f = np.fft.rfftfreq(pad, t[1] - t[0])
    R = np.fft.rfft(ref - ref[0], n=pad); T = np.fft.rfft(test - test[0], n=pad)
    m = (f >= band[0]) & (f <= band[1]) & (np.abs(R) > 0.05 * np.abs(R).max()) & (np.abs(T) > 0.05 * np.abs(T).max())
    dphi = np.unwrap(np.angle(T[m]) - np.angle(R[m]))
    w = np.abs(R[m]) * np.abs(T[m])
    A = np.stack([np.ones(m.sum()), 2 * np.pi * f[m]], axis=1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], dphi * np.sqrt(w), rcond=None)
    phi0, tau = coef
    phi0 = (np.degrees(phi0) + 180) % 360 - 180
    res = np.degrees(np.sqrt(np.average((dphi - A @ coef) ** 2, weights=w)))
    return phi0, tau * 1e12, res


def route2_trace(folder, X, n=3):
    meta = json.load(open(os.path.join("results", "tip_positions", folder, "meta.json")))
    Q = []
    for h in meta["heights_nm"]:
        d = np.load(os.path.join("results", "tip_positions", folder, f"X{X:+.1f}_h{int(h)}.npz")); Q.append(d["Q_tip"]); t = d["t"]
    dem = demodulate_positions(meta["heights_nm"], np.array(Q))
    return t, detected_waveform(t, dem[n], order=2)


def route1_trace(folder, X, n=3):
    d = np.load(os.path.join("results", "pulse_chain", folder, f"X{X:+.1f}.npz")); t = d["t"]
    return t, detected_waveform(t, d[f"S{n}"], order=2)


def best_shift(t, ref, test):
    """Time-domain check: best delay for same-sign and inverted templates, residual relative to the test RMS."""
    dt = t[1] - t[0]; out = {}
    for name, sgn in (("same sign", 1), ("inverted", -1)):
        best = (np.inf, 0)
        for k in range(-80, 81):
            r = np.roll(ref, k) * sgn
            a = np.dot(r, test) / np.dot(r, r)
            if a <= 0:                      # the template's sign is fixed by the variant; a negative fit is the other variant
                continue
            res = np.sqrt(np.mean((test - a * r) ** 2)) / np.sqrt(np.mean(test ** 2))
            if res < best[0]:
                best = (res, k * dt * 1e12)
        out[name] = dict(residual=round(float(best[0]), 3), shift_ps=round(float(best[1]), 3))
    return out


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    rows = []
    def add(label, tr, tt):
        phi0, tau, res = phase_fit(tr[0], tr[1], tt[1]); bs = best_shift(tr[0], tr[1], tt[1])
        rows.append((label, phi0, tau, res, bs))
    for route, fn, lab in (("route 1", lambda X: route1_trace("P2mW_notipmap_degen", X), "tip-free + FDM"),
                           ("route 2", lambda X: route2_trace("P2mW_sym", X), "grounded tip")):
        add(f"{route} ({lab}): +25 V X=+5  vs  +25 V X=-5  (= -25 V, X=+5: same X, polarity flipped)", fn(5.0), fn(-5.0))
        add(f"{route} ({lab}): +25 V X=+5  vs  +25 V X=+2.5", fn(5.0), fn(2.5))
        add(f"{route} ({lab}): +25 V X=+5  vs  +25 V X=0", fn(5.0), fn(0.0))
        add(f"{route} ({lab}): +25 V X=-5  vs  +25 V X=-2.5", fn(-5.0), fn(-2.5))
    if os.path.exists("results/tip_positions/P2mW_sym_Vtipm12/X+5.0_h370.npz"):
        p = lambda X: route2_trace("P2mW_sym_Vtip12", X); m = lambda X: route2_trace("P2mW_sym_Vtipm12", X)
        add("route 2, tip at a FIXED potential (+12.5 V vs midpoint): +25 V X=+5  vs  mirror pair -25 V X=-5", p(5.0), m(5.0))
        add("route 2, tip at a FIXED potential: +25 V X=-5  vs  mirror pair -25 V X=+5", p(-5.0), m(-5.0))
        add("route 2, tip at +12.5 V: +25 V X=+5 vs +25 V X=-5 (same X, polarity flipped, tip potential mirrored)", p(5.0), p(-5.0))
    print(f"{'pair':100s} {'phi0 (deg)':>10s} {'tau (ps)':>9s} {'res (deg)':>9s}   best shift (same / inverted)")
    for label, phi0, tau, res, bs in rows:
        print(f"{label:100s} {phi0:10.0f} {tau:9.2f} {res:9.0f}   {bs['same sign']}  /  {bs['inverted']}")
